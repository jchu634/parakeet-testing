import onnx
from pathlib import Path
for i in ["encoder-model.onnx", "decoder_joint-model.onnx"]:
    model = onnx.load(Path('nemo-onnx') / i)
    model.ir_version = 8
    model = onnx.save(model, Path('nemo-onnx-8') / i)
