import json
from onnx.external_data_helper import convert_model_to_external_data
import onnx
import shutil
import nemo.collections.asr as nemo_asr
from pathlib import Path


def test(model):
    output = model.transcribe(['2086-149220-0033.wav'])
    print(output[0].text)


def exportParakeetAsOnnxReference(model, onnx_dir):
    onnx_dir.mkdir(exist_ok=True)
    model.export(str(Path(onnx_dir, "model.onnx")))

    with Path(onnx_dir, "vocab.txt").open("wt") as f:
        for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
            f.write(f"{token} {i}\n")


def exportParakeetAsOnnxOptimised(model, onnx_dir):
    enable_local_attn = True
    # the ONNX model may have issues with long audios if chunking is enabled
    conv_chunking_factor = -1

    if enable_local_attn:
        model.change_attention_model('rel_pos_local_attn', [128, 128])  # local attn
        # Enable chunking for subsampling module
        # 1 = auto select, -1 = disabled, other values should be power of 2
        model.change_subsampling_conv_chunking_factor(conv_chunking_factor)

    shutil.rmtree(onnx_dir)
    onnx_temp_dir = onnx_dir / 'temp'
    onnx_temp_dir.mkdir(parents=True, exist_ok=True)

    model.export(str(Path(onnx_temp_dir, 'model.onnx')))

    encoder_onnx_file = onnx_temp_dir / 'encoder-model.onnx'
    data_file = encoder_onnx_file.name + '.data'
    onnx_model = onnx.load(encoder_onnx_file)

    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
        convert_attribute=False
    )

    onnx.save_model(
        onnx_model,
        onnx_dir / encoder_onnx_file.name,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
    )

    decoder_joiner_onnx_file = onnx_temp_dir / 'decoder_joint-model.onnx'
    decoder_joiner_onnx_file.rename(onnx_dir / decoder_joiner_onnx_file.name)

    shutil.rmtree(onnx_temp_dir)

    # vocab.txt
    with Path(onnx_dir, 'vocab.txt').open('wt') as f:
        for i, token in enumerate([*model.tokenizer.vocab, '<blk>']):
            f.write(f'{token} {i}\n')

    # config.json
    config_path = onnx_dir / 'config.json'
    config = {
        'model_type': 'nemo-conformer-tdt',
        'features_size': 128,
        'subsampling_factor': 8,
        'enable_local_attn': enable_local_attn,
        'conv_chunking_factor': conv_chunking_factor,
    }
    with open(config_path, 'w+') as fd:
        json.dump(config, fd)


if __name__ == "__main__":
    model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
    exportParakeetAsOnnxOptimised(model, Path('nemo-onnx'))
    for i in ["encoder-model.onnx", "decoder_joint-model.onnx"]:
        model = onnx.load(Path('nemo-onnx') / i, load_external_data=False)
        model.ir_version = 9
        model = onnx.save(model, Path('nemo-onnx-8') / i)
