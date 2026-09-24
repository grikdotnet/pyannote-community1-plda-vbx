# Model assets and provenance

`manifest.json` records the SHA-256 hash, source, license, graph contract, and
installed conversion tool version for every local model asset. The two NCNN
`.param` and `.bin` pairs are generated from the FP32 ONNX sources with
`python scripts/convert_models.py`. The script pins pnnx `20260526`, uses
`fp16=0`, and tests both normal and alternate input shapes through
`tests/test_neural_parity.py`.

Install the conversion tool separately with `uv pip install --no-deps
pnnx==20260526` in the development environment. This avoids pnnx's optional
Torch dependency; neither conversion from ONNX nor runtime inference uses it.

The [segmentation ONNX export](https://huggingface.co/FredrikKarlssonSpeech/pyannote-speaker-diarization-onnx)
and [split embedding encoder and projection](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split)
are CC-BY-4.0 derivatives of pyannote Community-1. Their exact downloaded
revisions are in the manifest. Runtime projection arrays are copied to
`models/resnet_seg_1_*.npy`, while the original split files remain in
`reference/pyannote-community-1-onnx-split/`. The user supplied PLDA files,
now at `models/plda.npz` and `models/xvec_transform.npz`, from
[Community-1 PLDA](https://huggingface.co/pyannote/speaker-diarization-community-1/tree/main/plda),
also CC-BY-4.0; their upstream revision was not recorded and remains marked
unresolved. This repository does not train or alter those source weights.

pnnx leaves four 1D instance normalization layers with a type NCNN cannot
load. `scripts/finish_segmentation_ncnn.py` maps them to native NCNN
`InstanceNorm` layers, reshapes their tensors, and corrects affine weight
order. The fixed graph matches the source ONNX scores at both verified shapes.

The PLDA score and VBx HMM equations in `src/diarization/` are adapted from
the local [BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx) reference
under Apache-2.0. Kaldi fbank features use
[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
under Apache-2.0. The runtime imports no source from `reference/vbx/`.

The segmentation frame timing, powerset ordering, and fbank settings were
cross-checked against the Community-1 export documentation and the
[pure ONNX diarization reference](https://github.com/welcomyou/sherpa-vietnamese-asr/blob/main/core/speaker_diarization_pure_ort.py).
