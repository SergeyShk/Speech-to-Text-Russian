# Models

## Default model

Speech is recognized by [GigaAM-v3](https://github.com/salute-developers/GigaAM) E2E RNN-T (`gigaam-v3-e2e-rnnt`) - an open Russian speech recognition model by SberDevices under the MIT license. It writes text with punctuation, capital letters and numbers as digits right away. The model runs through [onnx-asr](https://github.com/istupakov/onnx-asr) and ONNX Runtime, without PyTorch, with int8 weights by default.

## Other models

Any onnx-asr model can be chosen by name with `-m` or the `model` argument of `Transcriber`; it is downloaded from Hugging Face on the first run. Models that recognize Russian:

| Model | Features |
| :---- | :------- |
| `gigaam-v3-e2e-rnnt`, `gigaam-v3-e2e-ctc` | Punctuation, capital letters and numbers as digits; CTC is usually faster than RNN-T |
| `gigaam-v3-rnnt`, `gigaam-v3-ctc` | Lowercase without punctuation, numbers as words |
| `gigaam-v2-rnnt`, `gigaam-v2-ctc` | The previous GigaAM version |
| `t-tech/t-one` | T-Bank model for telephone speech |
| `nemo-fastconformer-ru-rnnt`, `nemo-fastconformer-ru-ctc` | NVIDIA NeMo FastConformer for Russian with punctuation |
| `alphacep/vosk-model-ru`, `alphacep/vosk-model-small-ru` | Vosk models; small is for low-end computers |
| `onnx-community/whisper-large-v3-turbo` | OpenAI Whisper multilingual model; much heavier than the others |
| `nemo-parakeet-tdt-0.6b-v3`, `nemo-canary-1b-v2` | NVIDIA multilingual models |

The full list and the model file formats are in the [onnx-asr documentation](https://github.com/istupakov/onnx-asr). Which model is best for your recordings is shown by a [quality evaluation](evaluation.md) on your own references:

``` bash
stt-ru evaluate eval.csv -m gigaam-v3-e2e-rnnt -m t-tech/t-one -m alphacep/vosk-model-ru
```

## Parameters { #options }

The model, device and speech detection parameters are shared by the `transcribe`, `evaluate` and `serve` commands.

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `-m`, `--model` | `gigaam-v3-e2e-rnnt` | onnx-asr model; can be repeated for `evaluate` |
| `--quantization` | `int8` | Weight quantization: `int8` or `none` - original weights |
| `--model-dir` | - | Folder with model files; for `evaluate` only with one model |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda`, `coreml`, `directml` (see [Installation](installation.md#devices)) |
| `--mono` | off | Mix channels into one |
| `--ffmpeg` | - | Path to the ffmpeg program |
| `--vad-threshold`, `--min-speech-ms`, `--max-speech-s`, `--min-silence-ms`, `--speech-pad-ms` | see [below](#vad) | Speech detection parameters |

## Quantization

int8 weights are about four times smaller than the original ones and usually faster on a CPU. `--quantization none` takes the original weights; a [quality evaluation](evaluation.md) with both values shows the difference on your recordings. If a model has no int8 weights, the original ones are loaded with a warning.

## Model files on disk { #local }

`--model-dir` (the `model_dir` argument) takes the model files from a folder: for example, on a computer without internet. If the folder does not exist, the model is downloaded into it. A model from the Hugging Face cache also works without network: the cache is set by the `HF_HOME` variable (see [Installation](installation.md#models)).

``` bash
stt-ru transcribe call.wav -m gigaam-v3-e2e-rnnt --model-dir /models/gigaam-v3-e2e-rnnt
```

## Speech detection { #vad }

!!! info ""
    **stt_ru.vad.VadOptions**

Before recognition the recording is cut into speech segments: the models accept segments of at most 20-30 s. Speech is detected by [Silero VAD](https://github.com/snakers4/silero-vad), and the defaults are those of Silero VAD and onnx-asr.

| Command parameter | `VadOptions` field | Default | Description |
| :---------------- | :----------------- | :------ | :---------- |
| `--vad-threshold` | `threshold` | `0.5` | Speech probability that starts a segment, from 0 to 1 |
| - | `neg_threshold` | `threshold - 0.15` | Probability below which a segment ends |
| `--min-speech-ms` | `min_speech_duration_ms` | `250` | Shortest segment, ms: shorter ones are dropped |
| `--max-speech-s` | `max_speech_duration_s` | `20` | Longest segment, s: longer ones are cut |
| `--min-silence-ms` | `min_silence_duration_ms` | `100` | Shortest pause that separates segments, ms |
| `--speech-pad-ms` | `speech_pad_ms` | `30` | Padding at the segment edges, ms |

*   A noisy recording gives segments of noise - raise `--vad-threshold`, for example to 0.6.
*   The model misses quiet speech - lower the threshold to 0.35-0.4.
*   Utterances are split into segments that are too short - increase `--min-silence-ms`, for example to 300: segments become longer and subtitles less frequent.
*   `max_speech_duration_s` must be at least one second longer than two `speech_pad_ms`.

``` python
>>> from stt_ru import Transcriber
>>> from stt_ru.vad import VadOptions

>>> transcriber = Transcriber(vad=VadOptions(threshold=0.6, min_silence_duration_ms=300))
```
