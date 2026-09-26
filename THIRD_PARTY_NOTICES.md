# Third-party notices

The root [`LICENSE`](LICENSE) is Apache License 2.0 for this project's own
software code. It does not relicense third-party code, model weights, converted
model files, audio, or annotations. Those materials retain their upstream
terms. Keep the applicable notices when redistributing them.

## Third-party code

| Local material | Source and applicable license | Use in this project |
| --- | --- | --- |
| `reference/vbx/` source files | [BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx), [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Reference implementation. The PLDA scoring and VBx HMM code in `src/diarization/plda.py` and `src/diarization/vbx.py` adapts its equations; these project files identify that source. Preserve upstream copyright notices in copied or modified files and mark modifications when distributing them. |
| `reference/pyannote-audio/` | [pyannote.audio](https://github.com/pyannote/pyannote-audio), MIT for its project code; see its [local license](reference/pyannote-audio/LICENSE) | Development reference. Some files carry additional or different notices, including [WeSpeaker's Apache-2.0 license](reference/pyannote-audio/src/pyannote/audio/models/embedding/wespeaker/LICENSE.WeSpeaker); retain the applicable file notices. |
| `reference/pyannote-community-1-onnx-split/split_pyannote_embedding.py` | [welcomyou/pyannote-community-1-onnx-split](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split), repository marked [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Reproducibility reference script. The root code license does not replace its source terms. |

The installed Python dependencies are distributed separately under their own
licenses. In particular, [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
is Apache-2.0 and supplies the runtime filterbank implementation; its source is
not copied into this project's `src/` tree.

## Model weights and converted model files

The weights and their converted forms are [CC BY 4.0](models/LICENSE), under
the terms of their respective upstream licensors. The local license text is
provided for these assets; it is **not** a new license grant by this project.
See [model provenance](models/README.md) and [the manifest](models/manifest.json)
for paths, sources, hashes, and known revisions.

| Local material | Source | Local processing |
| --- | --- | --- |
| `models/segmentation.param`, `models/segmentation.bin` | [Community-1 segmentation via FredrikKarlssonSpeech's ONNX export](https://huggingface.co/FredrikKarlssonSpeech/pyannote-speaker-diarization-onnx) | Converted ONNX to NCNN; corrected four unsupported instance normalization layers and affine weight order. |
| `models/embedding_encoder.param`, `models/embedding_encoder.bin`, `models/resnet_seg_1_weight.npy`, `models/resnet_seg_1_bias.npy` | [Community-1 split embedding encoder and projection](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split) | Converted the encoder from ONNX to NCNN; copied projection arrays without retraining. |
| `models/plda.npz`, `models/xvec_transform.npz` | [BUT Speech@FIT DiariZen PLDA](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md/tree/6285693ddd5b38e8229acb93f864f3d04a82bee1/plda), explicitly [CC BY 4.0 for these two files](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md/blob/6285693ddd5b38e8229acb93f864f3d04a82bee1/plda/LICENSE); also distributed with [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1/tree/main/plda) | Locally supplied copies match both upstream file SHA-256 hashes. No training or model alteration is documented. The pinned source revision is in the manifest. |

The original ONNX files remain at
`reference/FredrikKarlssonSpeech-pyannote-onnx/segmentation/model.onnx`,
`reference/FredrikKarlssonSpeech-pyannote-onnx/embedding/model.onnx`, and
`reference/pyannote-community-1-onnx-split/embedding_encoder.onnx`. They also
retain their upstream CC BY 4.0 terms. Attribute pyannote and the identified
export/split authors, link the source and
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/), and
describe conversions or other modifications when sharing these assets.

## Evaluation data

The AMI Meeting Corpus audio and annotations used under `reference/vbx/` are
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) according to the
[AMI corpus site](https://groups.inf.ed.ac.uk/ami/). Credit the AMI Meeting
Corpus, link its source and license, and identify any crops or annotation
changes when distributing them. The [fixture provenance note](docs/research/diarization-fixture-provenance.md)
records the known origin and limits of `ES2005a`. Other materials in the VBx
reference tree may have separate source terms; the root code license does not
cover them.
