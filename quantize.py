import json
import onnx
import shutil
import onnxruntime as rt
import numpy as np
import wave
from pathlib import Path
import os
import io

from pydub import AudioSegment

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config, DEFAULT_CONFIG_MAPPING

# --- Utility functions ---


def read_file(filename: str):
    fileObject = filename
    # If input is not WAV, convert via pydub
    if not filename.lower().endswith(".wav"):
        audio = AudioSegment.from_file(filename)
        fileObject = io.BytesIO()
        audio.export(fileObject, format="wav")
        fileObject.seek(0)

    with wave.open(fileObject, mode="rb") as f:
        data = f.readframes(f.getnframes())
        zero_value = 0
        if f.getsampwidth() == 1:
            buffer = np.frombuffer(data, dtype="u1")
            zero_value = 1
        elif f.getsampwidth() == 3:
            buffer = np.zeros((len(data) // 3, 4), dtype="V1")
            buffer[:, -3:] = np.frombuffer(data, dtype="V1").reshape(-1, f.getsampwidth())
            buffer = buffer.view(dtype="<i4")
        else:
            buffer = np.frombuffer(data, dtype=f"<i{f.getsampwidth()}")
        max_value = 2 ** (8 * buffer.itemsize - 1)
        arr = buffer.reshape(f.getnframes(), f.getnchannels()).astype(np.float32) / max_value - zero_value
        if arr.shape[1] != 1:
            raise ValueError("Only mono audio supported")
        return arr[:, 0], f.getframerate()


def pad_list(arrays):
    lens = np.array([array.shape[0] for array in arrays], dtype=np.int64)
    result = np.zeros((len(arrays), lens.max()), dtype=np.float32)
    for i, x in enumerate(arrays):
        result[i, : x.shape[0]] = x[: min(x.shape[0], result.shape[1])]
    return result, lens


# --- Calibration Data Reader for ASR Preprocessor (to feed float preprocessor) ---
class ASRPreprocessorCalibrationDataReader:
    def __init__(self, audio_data_folder: str, batch_size: int = 1):
        super().__init__()
        self.audio_files = []
        for filename in os.listdir(audio_data_folder):
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                self.audio_files.append(os.path.join(audio_data_folder, filename))

        if not self.audio_files:
            raise ValueError(f"No audio files found in calibration data folder: {audio_data_folder}")

        self.batch_size = batch_size
        self.current_batch_index = 0
        self.all_processed_batches = []
        self._load_and_preprocess_audio_into_batches()

    def _load_and_preprocess_audio_into_batches(self):
        waveforms_buffer = []

        for i, audio_path in enumerate(self.audio_files):
            try:
                waveform, sample_rate = read_file(audio_path)
                if sample_rate != 16000:
                    print(f"Warning: Audio file {audio_path} has sample rate {sample_rate}Hz. "
                          "Model expects 16kHz. Skipping this file for calibration.")
                    continue
                waveforms_buffer.append(waveform)

                if len(waveforms_buffer) == self.batch_size:
                    padded_waveforms, padded_waveforms_lens = pad_list(waveforms_buffer)
                    self.all_processed_batches.append({
                        "waveforms": padded_waveforms,
                        "waveforms_lens": padded_waveforms_lens
                    })
                    waveforms_buffer = []  # Reset buffer for next batch
            except Exception as e:
                print(f"Error processing {audio_path} for preprocessor calibration: {e}")
                continue

        # Add any remaining waveforms as a final batch
        if waveforms_buffer:
            padded_waveforms, padded_waveforms_lens = pad_list(waveforms_buffer)
            self.all_processed_batches.append({
                "waveforms": padded_waveforms,
                "waveforms_lens": padded_waveforms_lens
            })

        if not self.all_processed_batches:
            raise ValueError("No valid 16kHz audio data could be processed into batches for preprocessor calibration.")

    def get_next(self):
        if self.current_batch_index < len(self.all_processed_batches):
            input_data = self.all_processed_batches[self.current_batch_index]
            self.current_batch_index += 1
            return input_data
        return None

# --- Calibration Data Reader for ASR Encoder ---


class ASREncoderCalibrationDataReader:
    def __init__(self, preprocessor_onnx_path: str, audio_data_folder: str, onnx_options, batch_size: int = 1):
        super().__init__()
        # Use a float preprocessor session to generate accurate calibration data for the encoder
        self.preprocessor_session = rt.InferenceSession(str(preprocessor_onnx_path), **onnx_options)
        self.preprocessor_reader = ASRPreprocessorCalibrationDataReader(audio_data_folder, batch_size=batch_size)
        self.processed_data_cache = []
        self.index = 0
        self._load_and_preprocess_audio_for_encoder()

    def _load_and_preprocess_audio_for_encoder(self):
        # Generate encoder inputs by running the float preprocessor with calibration data
        while True:
            preprocessor_input = self.preprocessor_reader.get_next()
            if preprocessor_input is None:
                break

            try:
                features, features_lens = self.preprocessor_session.run(
                    ["features", "features_lens"], preprocessor_input
                )
                # Encoder expects "audio_signal" and "length"
                self.processed_data_cache.append({
                    "audio_signal": features,
                    "length": features_lens
                })
            except Exception as e:
                print(f"Error running preprocessor for encoder calibration: {e}")
                continue

        if not self.processed_data_cache:
            raise ValueError("No valid data could be generated for encoder calibration.")

    def get_next(self):
        if self.index < len(self.processed_data_cache):
            input_data = self.processed_data_cache[self.index]
            self.index += 1
            return input_data
        return None

# --- Calibration Data Reader for ASR Decoder/Joint ---


class ASRDecoderJointCalibrationDataReader:
    def __init__(self,
                 encoder_onnx_path: str,
                 decoder_joint_onnx_path: str,
                 preprocessor_onnx_path: str,
                 audio_data_folder: str,
                 config_file_path: str,  # To get vocab info
                 vocab_file_path: str,  # To get vocab info
                 onnx_options,
                 num_calibration_steps_per_audio: int = 5,
                 ):
        super().__init__()
        self.num_calibration_steps_per_audio = num_calibration_steps_per_audio

        # Initialize the float preprocessor and encoder to generate encoder_outputs
        self.preprocessor_session = rt.InferenceSession(str(preprocessor_onnx_path), **onnx_options)
        self.encoder_session = rt.InferenceSession(str(encoder_onnx_path), **onnx_options)
        self.decoder_joint_session = rt.InferenceSession(
            str(decoder_joint_onnx_path), **onnx_options)  # For getting input shapes

        # We'll use a batch_size of 1 for the preprocessor reader when simulating decode steps
        # to ensure we process one audio file at a time, mimicking autoregressive inference.
        self.preprocessor_reader = ASRPreprocessorCalibrationDataReader(audio_data_folder, batch_size=1)

        self.processed_data_cache = []
        self.index = 0

        # Load vocab and config to get necessary model parameters
        with open(config_file_path, "rt", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(vocab_file_path, "rt", encoding="utf-8") as f:
            tokens = {token: int(id) for token, id in (line.strip("\n").split(" ") for line in f.readlines())}
        self._vocab_size = len(tokens)
        self._blank_idx = tokens["<blk>"]

        self._max_tokens_per_step = self.config.get("max_tokens_per_step", 10)

        # Get input state shapes from decoder_joint_session
        shapes = {x.name: x.shape for x in self.decoder_joint_session.get_inputs()}
        self._state1_shape = (shapes["input_states_1"][0], 1, shapes["input_states_1"][2])
        self._state2_shape = (shapes["input_states_2"][0], 1, shapes["input_states_2"][2])

        self._simulate_decoder_steps_for_calibration()

    def _create_initial_state(self):
        return (
            np.zeros(shape=self._state1_shape, dtype=np.float32),
            np.zeros(shape=self._state2_shape, dtype=np.float32),
        )

    def _simulate_decoder_steps_for_calibration(self):
        while True:
            preprocessor_input = self.preprocessor_reader.get_next()
            if preprocessor_input is None:
                break

            try:
                # 1. Run Preprocessor (float)
                features, features_lens = self.preprocessor_session.run(
                    ["features", "features_lens"], preprocessor_input
                )

                # 2. Run Encoder (float)
                encoder_out, encoder_out_lens = self.encoder_session.run(
                    ["outputs", "encoded_lengths"], {"audio_signal": features, "length": features_lens}
                )
                encoder_out = encoder_out.transpose(0, 2, 1)[0]  # (hidden, seq_len) from (1, seq_len, hidden)
                encoder_out_len = encoder_out_lens[0]

                prev_state = self._create_initial_state()
                tokens = []

                t = 0
                emitted_tokens_count = 0

                # Simulate a few decode steps to gather diverse inputs for calibration
                for _ in range(self.num_calibration_steps_per_audio):
                    if t >= encoder_out_len:
                        break  # End of audio features

                    # Prepare inputs for decoder_joint
                    decoder_joint_inputs = {
                        "encoder_outputs": encoder_out[None, :, None, t],
                        "targets": np.array([[tokens[-1] if tokens else self._blank_idx]], dtype=np.int64),
                        "target_length": np.array([1], dtype=np.int64),
                        "input_states_1": prev_state[0],
                        "input_states_2": prev_state[1],
                    }

                    # Append this input dictionary to our cache
                    self.processed_data_cache.append(decoder_joint_inputs)

                    # Simulate one step of decoding to get the *next* state and token for the next calibration sample
                    outputs, state1_next, state2_next = self.decoder_joint_session.run(
                        ["outputs", "output_states_1", "output_states_2"], decoder_joint_inputs
                    )

                    output = np.squeeze(outputs)
                    probs = output[: self._vocab_size]
                    step = int(output[self._vocab_size:].argmax())
                    token = probs.argmax()

                    if token != self._blank_idx:
                        prev_state = (state1_next, state2_next)
                        tokens.append(int(token))
                        emitted_tokens_count += 1

                    if step > 0:
                        t += step
                        emitted_tokens_count = 0
                    elif token == self._blank_idx or emitted_tokens_count == self._max_tokens_per_step:
                        t += 1
                        emitted_tokens_count = 0

            except Exception as e:
                print(
                    f"Error simulating decoder for audio from {preprocessor_input.get('waveforms_lens', 'N/A')} for decoder calibration: {e}")
                continue

        if not self.processed_data_cache:
            raise ValueError("No valid data could be generated for decoder_joint calibration.")

    def get_next(self):
        if self.index < len(self.processed_data_cache):
            input_data = self.processed_data_cache[self.index]
            self.index += 1
            return input_data
        return None


# --- Main Quantization Script ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize NemoConformerTdt ONNX ASR models (Encoder and Decoder/Joint)")
    parser.add_argument(
        "folder", help="Path to folder with model files (encoder-model.onnx, decoder_joint-model.onnx, vocab.txt, config.json, nemo*.onnx)")
    parser.add_argument("calibration_data_folder",
                        help="Path to a folder containing mono 16kHz audio files for calibration.")
    parser.add_argument("--output_folder", default="quantized_models", help="Folder to save quantized models.")
    quant_configs = [
        'UINT8_DYNAMIC_QUANT',
        'XINT8',
        'XINT8_ADAROUND',
        'XINT8_ADAQUANT',
        'S8S8_AAWS',
        'S8S8_AAWS_ADAROUND',
        'S8S8_AAWS_ADAQUANT',
        'U8S8_AAWS',
        'U8S8_AAWS_ADAROUND',
        'U8S8_AAWS_ADAQUANT',
        'U8U8_AAWA',
        'S16S8_ASWS',
        'S16S8_ASWS_ADAROUND',
        'S16S8_ASWS_ADAQUANT',
        'A8W8',
        'A8W8_ADAROUND',
        'A8W8_ADAQUANT',
        'A16W8',
        'A16W8_ADAROUND',
        'A16W8_ADAQUANT',
        'U16S8_AAWS',
        'U16S8_AAWS_ADAROUND',
        'U16S8_AAWS_ADAQUANT',
        'FP16',
        'FP16_ADAQUANT',
        'BF16',
        'BF16_ADAQUANT',
        'BFP16',
        'BFP16_ADAQUANT',
        'MX4',
        'MX4_ADAQUANT',
        'MX6',
        'MX6_ADAQUANT',
        'MX9',
        'MX9_ADAQUANT',
        'MXFP8E5M2',
        'MXFP8E5M2_ADAQUANT',
        'MXFP8E4M3',
        'MXFP8E4M3_ADAQUANT',
        'MXFP6E3M2',
        'MXFP6E3M2_ADAQUANT',
        'MXFP6E2M3',
        'MXFP6E2M3_ADAQUANT',
        'MXFP4E2M1',
        'MXFP4E2M1_ADAQUANT',
        'MXINT8',
        'MXINT8_ADAQUANT',
        'S16S16_MIXED_S8S8',
        'BF16_MIXED_BFP16',
        'BF16_MIXED_BFP16_ADAQUANT',
        'BF16_MIXED_MXINT8',
        'BF16_MIXED_MXINT8_ADAQUANT',
        'BF16_BFP16',
        'BF16_MXINT8',
        'MX9_INT8',
        # configs for amateur
        'INT8_CNN_DEFAULT',
        'INT16_CNN_DEFAULT',
        'INT8_TRANSFORMER_DEFAULT',
        'INT16_TRANSFORMER_DEFAULT',
        'INT8_CNN_ACCURATE',
        'INT16_CNN_ACCURATE',
        'INT8_TRANSFORMER_ACCURATE',
        'INT16_TRANSFORMER_ACCURATE',
        'MATMUL_NBITS',
    ]
    parser.add_argument(
        "--config",
        "-c",
        choices=quant_configs,
        default="default",
        help=f"Choose a quantization configuration option: {', '.join(quant_configs)} (default: A8W8)",
    )

    args = parser.parse_args()

    model_folder = Path(args.folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Explicitly set providers to only CPUExecutionProvider to avoid TensorRT/CUDA warnings
    onnx_options = {"providers": ["CPUExecutionProvider"]}

    print(f"\n--- Using default config {args.config} quantization configuration ---")
    quantization_config = Config(global_quant_config=get_default_config(args.config))

    # --- Identify the Float Preprocessor ---
    preproc_files = list(model_folder.glob("nemo*.onnx"))
    if not preproc_files:
        raise FileNotFoundError("No preprocessor ONNX file found in model folder (e.g., nemo80.onnx). "
                                "This float preprocessor is needed to generate calibration data for other models.")
    preprocessor_path = preproc_files[0]
    print(f"\n--- Using Float Preprocessor: {preprocessor_path.name} ---")

    # --- 1. Quantize the Encoder Model (encoder-model.onnx) ---
    encoder_path = model_folder / "encoder-model.onnx"

    print(f"\n--- Quantizing Encoder: {encoder_path.name} ---")
    encoder_calib_reader = ASREncoderCalibrationDataReader(
        preprocessor_path,  # Use preprocessor to generate encoder calibration data
        args.calibration_data_folder,
        onnx_options,
        batch_size=1
    )
    encoder_quantizer = ModelQuantizer(quantization_config)
    quantized_encoder_path = output_folder / f"{encoder_path.stem}_quantized.onnx"

    encoder_quantizer.quantize_model(
        str(encoder_path),
        str(quantized_encoder_path),
        encoder_calib_reader
    )
    print(f"Quantized encoder saved to: {quantized_encoder_path}")

    # --- 2. Quantize the Decoder/Joint Model (decoder_joint-model.onnx) ---
    decoder_joint_path = model_folder / "decoder_joint-model.onnx"
    config_path = model_folder / "config.json"
    vocab_path = model_folder / "vocab.txt"

    if decoder_joint_path.exists() and config_path.exists() and vocab_path.exists():
        print(f"\n--- Quantizing Decoder/Joint: {decoder_joint_path.name} ---")
        decoder_joint_calib_reader = ASRDecoderJointCalibrationDataReader(
            encoder_onnx_path=encoder_path,              # Use the encoder here to get accurate intermediate activations
            decoder_joint_onnx_path=decoder_joint_path,  # And the decoder for simulation
            preprocessor_onnx_path=preprocessor_path,
            audio_data_folder=args.calibration_data_folder,
            config_file_path=config_path,
            vocab_file_path=vocab_path,
            onnx_options=onnx_options,
            num_calibration_steps_per_audio=10,
        )
        decoder_joint_quantizer = ModelQuantizer(quantization_config)
        quantized_decoder_joint_path = output_folder / f"{decoder_joint_path.stem}_quantized.onnx"

        decoder_joint_quantizer.quantize_model(
            str(decoder_joint_path),
            str(quantized_decoder_joint_path),
            decoder_joint_calib_reader
        )
        print(f"Quantized decoder_joint saved to: {quantized_decoder_joint_path}")
    else:
        print(f"\n--- Skipping quantization of Decoder/Joint model ---")
        print(
            f"Required files not found: {decoder_joint_path.exists()=}, {config_path.exists()=}, {vocab_path.exists()=}")
        print("Please ensure 'decoder_joint-model.onnx', 'config.json', and 'vocab.txt' are in the model folder.")

    print("\nQuantization process completed.")
