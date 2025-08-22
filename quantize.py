from inference import NemoConformerTdt
import json
import string
import re
from inference import read_file
from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx import ModelQuantizer
import os

calibration_folder = "testing-data"
model_input_name = "parakeet-v3-0.6b"
model_folder = "nemo-onnx-8"
quantized_model_path = "nemo-onnx-bf16"


class CalibrationDataReader:
    def __init__(self, calib_data_folder: str, model_input_name: str):
        super().__init__()
        self.input_name = model_input_name
        self.processed_data = []
        self.data = self._load_calibration_data(calib_data_folder)
        self.index = 0

    def _load_calibration_data(self, data_folder: str):
        for audio_filename in os.listdir(data_folder):
            if audio_filename.lower().endswith(('.flac', '.wav')):
                audio_path = os.path.join(data_folder, audio_filename)
                audio = read_file(audio_path)
                self.processed_data.append(audio)
        return self.processed_data

    def get_next(self):
        if self.index < len(self.processed_data):
            input_data = {self.input_name: self.processed_data[self.index]}
            self.index += 1
            return input_data
        return None


def normalize_text(text):
    # Remove punctuation, lowercase, and collapse whitespace
    text = text.upper()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_calibration_answers(path):
    answers = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                fname, text = parts
                answers[fname] = text
    return answers


def evaluate_model(model_folder, answers_path, data_folder, output_log_path):
    model = NemoConformerTdt(model_folder, {"providers": ["CPUExecutionProvider"]})
    answers = load_calibration_answers(answers_path)
    results = []
    total = 0
    errors = 0
    for fname, ref_text in answers.items():
        # Find corresponding audio file (try .flac, then .wav)
        for ext in (".flac", ".wav"):
            audio_path = os.path.join(data_folder, fname + ext)
            if os.path.exists(audio_path):
                break
        else:
            results.append({"file": fname, "error": "Missing audio file"})
            continue
        try:
            waveform, sample_rate = read_file(audio_path)
            pred = model.recognize(waveform, sample_rate)
        except Exception as e:
            results.append({"file": fname, "error": str(e)})
            continue
        norm_pred = normalize_text(pred)
        norm_ref = normalize_text(ref_text)
        correct = norm_pred == norm_ref
        results.append({
            "file": fname,
            "prediction": pred,
            "reference": ref_text,
            "normalized_prediction": norm_pred,
            "normalized_reference": norm_ref,
            "match": correct
        })
        total += 1
        if not correct:
            errors += 1
    wer = errors / total if total else 0.0
    with open(output_log_path, "w", encoding="utf-8") as f:
        json.dump({"wer": wer, "total": total, "errors": errors, "results": results}, f, indent=2)
    print(
        f"Evaluation complete. WER (string match): {wer:.3f} ({errors}/{total} errors). Log written to {output_log_path}")


calib_data_reader = CalibrationDataReader(calibration_folder, model_input_name)

quant_config = get_default_config("BF16")
# quant_config.extra_options["BF16QDQToCast"] = True
config = Config(global_quant_config=quant_config)
quantizer = ModelQuantizer(config)

encoder_path = model_folder + "/encoder-model.onnx"
decoder_joint_path = model_folder + "/decoder_joint-model.onnx"

quantized_encoder_path = quantized_model_path + "/encoder-model.onnx"
quantized_decoder_joint_path = quantized_model_path + "/decoder_joint-model.onnx"


quantizer.quantize_model(encoder_path, quantized_encoder_path, calib_data_reader)
quantizer.quantize_model(decoder_joint_path, quantized_decoder_joint_path, calib_data_reader)


evaluate_model(
    quantized_model_path,
    os.path.join(calibration_folder, "calibration_full_answers.txt"),
    calibration_folder,
    "quantization_eval_log.json"
)
