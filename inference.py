import wave
import numpy as np
import onnxruntime as rt
from pathlib import Path
import json

from pydub import AudioSegment
import io

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

# --- Preprocessor ---


class Preprocessor:
    def __init__(self, onnx_bytes, onnx_options):
        self._preprocessor = rt.InferenceSession(onnx_bytes, **onnx_options)

    def __call__(self, waveforms, waveforms_lens):
        features, features_lens = self._preprocessor.run(
            ["features", "features_lens"], {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
        )
        return features, features_lens

# --- Model ---


class NemoConformerTdt:
    def __init__(self, folder, onnx_options):
        folder = Path(folder)
        encoder_path = folder / "encoder-model.onnx"
        decoder_joint_path = folder / "decoder_joint-model.onnx"
        vocab_path = folder / "vocab.txt"
        config_path = folder / "config.json"
        # Find preprocessor file (e.g. nemo80.onnx, nemo128.onnx, etc)
        preproc_files = list(folder.glob("nemo*.onnx"))
        if not preproc_files:
            raise FileNotFoundError("No preprocessor ONNX file found in folder")
        preprocessor_path = preproc_files[0]
        with open(config_path, "rt", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(preprocessor_path, "rb") as f:
            preprocessor_bytes = f.read()
        self._encoder = rt.InferenceSession(str(encoder_path), **onnx_options)
        self._decoder_joint = rt.InferenceSession(str(decoder_joint_path), **onnx_options)
        self._preprocessor = Preprocessor(preprocessor_bytes, onnx_options)
        with open(vocab_path, "rt", encoding="utf-8") as f:
            tokens = {token: int(id) for token, id in (line.strip("\n").split(" ") for line in f.readlines())}
        self._vocab = {id: token.replace("\u2581", " ") for token, id in tokens.items()}
        self._vocab_size = len(self._vocab)
        self._blank_idx = tokens["<blk>"]
        self._subsampling_factor = self.config.get("subsampling_factor", 8)
        self._max_tokens_per_step = self.config.get("max_tokens_per_step", 10)
        self.window_size = 0.01

    def _encode(self, features, features_lens):
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"], {"audio_signal": features, "length": features_lens}
        )
        return encoder_out.transpose(0, 2, 1), encoder_out_lens

    def _create_state(self):
        shapes = {x.name: x.shape for x in self._decoder_joint.get_inputs()}
        return (
            np.zeros(shape=(shapes["input_states_1"][0], 1, shapes["input_states_1"][2]), dtype=np.float32),
            np.zeros(shape=(shapes["input_states_2"][0], 1, shapes["input_states_2"][2]), dtype=np.float32),
        )

    def _decode(self, prev_tokens, prev_state, encoder_out):
        outputs, state1, state2 = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encoder_out[None, :, None],
                "targets": [[prev_tokens[-1] if prev_tokens else self._blank_idx]],
                "target_length": [1],
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            },
        )
        output = np.squeeze(outputs)
        state = (state1, state2)
        return output[: self._vocab_size], int(output[self._vocab_size:].argmax()), state

    def recognize(self, waveform, sample_rate=16000):
        waveforms, waveforms_lens = pad_list([waveform])
        features, features_lens = self._preprocessor(waveforms, waveforms_lens)
        encoder_out, encoder_out_lens = self._encode(features, features_lens)
        results = []
        for encodings, encodings_len in zip(encoder_out, encoder_out_lens):
            prev_state = self._create_state()
            tokens = []
            timestamps = []
            t = 0
            emitted_tokens = 0
            while t < encodings_len:
                probs, step, state = self._decode(tokens, prev_state, encodings[t])
                token = probs.argmax()
                if token != self._blank_idx:
                    prev_state = state
                    tokens.append(int(token))
                    timestamps.append(t)
                    emitted_tokens += 1
                if step > 0:
                    t += step
                    emitted_tokens = 0
                elif token == self._blank_idx or emitted_tokens == self._max_tokens_per_step:
                    t += 1
                    emitted_tokens = 0
            text = "".join([self._vocab[i] for i in tokens]).replace("  ", " ").strip()
            results.append(text)
        return results[0] if len(results) == 1 else results


# --- Main ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parakeet v3 ONNX ASR inference (single folder)")
    parser.add_argument(
        "folder", help="Path to folder with model files (encoder-model.onnx, decoder_joint-model.onnx, vocab.txt, config.json, nemo*.onnx)")
    parser.add_argument("file", help="Path to mono input file (16kHz)")
    args = parser.parse_args()
    onnx_options = {"providers": rt.get_available_providers()}
    waveform, sample_rate = read_file(args.file)
    if sample_rate != 16000:
        raise ValueError("Only 16kHz sample rate supported for this model")
    model = NemoConformerTdt(args.folder, onnx_options)
    text = model.recognize(waveform)
    print(text)
