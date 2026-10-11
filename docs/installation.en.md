# Installation

The library and the `stt-ru` command can be installed with `pip` or from the project repository with `git`. The ready-made Docker image is described on the [Docker](docker.md) page.

## Dependencies

*   `python` - 3.11 or newer
*   [`onnx-asr`](https://github.com/istupakov/onnx-asr) - 0.12, with ONNX Runtime for CPU and model downloads from Hugging Face
*   `numpy`
*   `soundfile`

## Installing with `pip`

``` bash
pip install stt-ru
```

Additional features come as extras:

| Extra | What it adds | Packages |
| :---- | :----------- | :------- |
| `ffmpeg` | M4A, AAC, AMR, WMA, the sound of videos and resampling without an installed ffmpeg | `imageio-ffmpeg` |
| `service` | The [web interface and REST API](service.md): the `stt-ru serve` command | `fastapi`, `uvicorn`, `python-multipart` |
| `eval` | [Quality evaluation](evaluation.md): the `stt-ru evaluate` command | `jiwer` |

``` bash
pip install "stt-ru[ffmpeg,service,eval]"
```

Or with [uv](https://docs.astral.sh/uv/):

``` bash
uv add "stt-ru[ffmpeg]"
```

!!! tip "ffmpeg"
    WAV, FLAC, OGG (including Opus), MP3, AIFF, AU, CAF and W64 are read by the soundfile library. M4A, AAC, AMR, WMA, the sound of videos, recordings with a sample rate the model does not accept and files without the recording length in the header need ffmpeg. stt-ru looks for it in `PATH`, then in the `imageio-ffmpeg` package from the `ffmpeg` extra; the path can be set with `--ffmpeg`. If ffmpeg is needed but not found, the file is not transcribed and the message says what to install.

## Models { #models }

A model is downloaded from Hugging Face on the first transcription and kept in its cache: the default model `gigaam-v3-e2e-rnnt` in int8 quantization together with Silero VAD takes about 230 MB. The cache is in `~/.cache/huggingface`; the `HF_HOME` environment variable sets another folder:

``` bash
export HF_HOME=/data/models
```

A model from the cache works without network. For a computer without internet, copy the whole cache folder to it - it holds both the model and Silero VAD - and point `HF_HOME` to it (see also [Models](models.md#local)).

## GPUs and other devices { #devices }

By default the model runs on the CPU with the `onnxruntime` package. For other devices it is replaced with another ONNX Runtime build: of the same version for NVIDIA GPUs, the latest one for DirectML (onnx-asr works with ONNX Runtime 1.18.1 and newer):

| Device | `--device` | Package | Requirements |
| :----- | :--------- | :------ | :----------- |
| NVIDIA GPU | `cuda` | `onnxruntime-gpu` | CUDA 13 and cuDNN 9 for ONNX Runtime 1.31 |
| DirectML (Windows) | `directml` | `onnxruntime-directml` | Windows 10 or newer |
| Apple CoreML | `coreml` | `onnxruntime` for macOS | explicit only: CoreML is slower than CPU on these models |

Replacing it for an NVIDIA GPU:

``` bash
pip show onnxruntime | grep Version    # the version, for example 1.31.0
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.31.0
```

The driver and library requirements of ONNX Runtime builds are in the [ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/). By default (`--device auto`) stt-ru takes the available devices in the ONNX Runtime priority order - GPU before CPU - except CoreML and TensorRT; an unavailable device requested explicitly is an error that says which package to install. There is a ready-made [Docker image](docker.md#gpu) for NVIDIA GPUs.

## Installing with `git`

The latest version from the project repository:

``` bash
git clone https://github.com/SergeyShk/Speech-to-Text-Russian.git
cd Speech-to-Text-Russian
pip install ".[ffmpeg,service,eval]"
```

For development [uv](https://docs.astral.sh/uv/) is more convenient: it creates an environment with all dependencies and development tools.

``` bash
uv sync --all-groups --all-extras
```

The workflow and the checks before a pull request are described in [CONTRIBUTING.md](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/CONTRIBUTING.md).
