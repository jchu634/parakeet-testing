## Parakeet Testing
My attempt at quantizing the NVIDIA Parakeet v3 models using AMD Quark for the Ryzen AI NPU compute engine.

Currently non-functional due to issues quantizing encoder model.\
Specifically: 
`[QUARK-WARNING]: Fail to Simplify ONNX model because The model does not have an ir_version set properly.`

Export scripts and the preprocessor is based on the ONNX-ASR Library.