import onnx
from pathlib import Path
model_folder = Path("nemo-onnx")
encoder_model = onnx.load(model_folder / "encoder-model.onnx")
print(f"encoder-model IR Version: {encoder_model.ir_version}")
print(f"encoder-model Opset Import: {[opset.version for opset in encoder_model.opset_import]}")

decoder_model = onnx.load(model_folder / "decoder_joint-model.onnx")
print(f"decoder-model IR Version: {decoder_model.ir_version}")
print(f"decoder-model Opset Import: {[opset.version for opset in decoder_model.opset_import]}")
