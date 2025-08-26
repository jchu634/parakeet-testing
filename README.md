## Parakeet Testing
My attempt at quantizing the NVIDIA Parakeet v3 models using AMD Quark for the Ryzen AI NPU compute engine.

Currently non-functional due to issues quantizing encoder model.\
Specifically: 
`[QUARK-WARNING]: Fail to Simplify ONNX model because The model does not have an ir_version set properly.`

Export scripts and the preprocessor is based on the ONNX-ASR Library.

### Notes:
uv will needed to be initialised with pip
- For new envs
    ```
    uv venv --seed
    ```
- For existing envs
    ```
    uv pip install pip
    ```

AMD-Quark may not install properly with ```uv add```
```
Failed to download `amd-quark==0.9`
  ├─▶ Failed to extract archive: amd_quark-0.9-py3-none-any.whl
  ╰─▶ ZIP file contains multiple entries with different contents for: amd_quark-0.9.dist-info/RECORD
```
To install AMD-QUARK run 
```
uv run python -m pip install amd-quark
```