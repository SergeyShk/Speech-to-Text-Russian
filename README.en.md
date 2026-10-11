<p align="center">
  <img src="https://raw.githubusercontent.com/SergeyShk/Speech-to-Text-Russian/master/docs/img/stt-ru.png" alt="stt-ru" width="480">
</p>

<h1 align="center">stt-ru</h1>

<p align="center">
  <b>Speech-to-Text Russian</b> - Russian speech recognition: a Python library, a command line tool, a web service and a Docker image
</p>

<p align="center">
  <a href="https://sergeyshk.github.io/Speech-to-Text-Russian/en/">Documentation</a> ·
  <a href="https://pypi.org/project/stt-ru/">PyPI</a> ·
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/pkgs/container/stt-ru">Docker</a> ·
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/README.md">Русский</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/stt-ru/"><img src="https://img.shields.io/pypi/v/stt-ru?logo=pypi&logoColor=FFE873" alt="Version"></a>
  <a href="https://pypi.org/project/stt-ru/"><img src="https://img.shields.io/pypi/pyversions/stt-ru.svg?logo=python&logoColor=FFE873" alt="Supported Python versions"></a>
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/actions/workflows/ci.yml"><img src="https://github.com/SergeyShk/Speech-to-Text-Russian/actions/workflows/ci.yml/badge.svg" alt="Build"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/sergeyshk/speech-to-text-russian.svg" alt="License"></a>
</p>

---

**stt-ru** recognizes Russian speech in recordings of conversations, lectures and dictation. Speech is recognized by the [GigaAM-v3](https://github.com/salute-developers/GigaAM) model through [onnx-asr](https://github.com/istupakov/onnx-asr): the text comes with punctuation and digits, the channels of a stereo recording are recognized as separate speakers, and the result is saved as subtitles and tables. The model runs on a CPU without PyTorch and on NVIDIA GPUs through ONNX Runtime.

* **[The transcribe command](https://sergeyshk.github.io/Speech-to-Text-Russian/en/transcribe/)** - files and folders, results of every file and one summary file per batch, monitoring a folder for new recordings
* **[Output formats](https://sergeyshk.github.io/Speech-to-Text-Russian/en/formats/)** - TXT, JSON, JSONL, SRT, VTT, ASS for Aegisub and CSV for spreadsheets
* **[Python library](https://sergeyshk.github.io/Speech-to-Text-Russian/en/python/)** - transcription of files and in-memory recordings, exporting the result
* **[Models](https://sergeyshk.github.io/Speech-to-Text-Russian/en/models/)** - any onnx-asr model: GigaAM, T-one, Vosk, Whisper and others; devices and speech detection parameters
* **[Quality evaluation](https://sergeyshk.github.io/Speech-to-Text-Russian/en/evaluation/)** - WER, CER and speed of models on your own references
* **[Web service](https://sergeyshk.github.io/Speech-to-Text-Russian/en/service/)** - a web interface with a waveform and navigation by segments, a REST API, a queue and an access token
* **[Docker](https://sergeyshk.github.io/Speech-to-Text-Russian/en/docker/)** - CPU and NVIDIA GPU images with the model inside

## Installation

Python 3.11 or newer is required.

```bash
pip install stt-ru
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add stt-ru
```

Extras: `ffmpeg` - M4A, AAC and the sound of videos without an installed ffmpeg; `service` - the web interface and REST API; `eval` - quality evaluation. All at once:

```bash
pip install "stt-ru[ffmpeg,service,eval]"
```

The model (about 230 MB) is downloaded from Hugging Face on the first run. GPUs and working without internet are covered on the [Installation](https://sergeyshk.github.io/Speech-to-Text-Russian/en/installation/) page.

## Quick start

```bash
stt-ru transcribe call.wav                                       # text to the console
stt-ru transcribe calls/ -o out/ -f srt,ass,csv --summary out/all.csv
stt-ru transcribe calls/ -o out/ --watch                         # folder monitoring
stt-ru evaluate eval.csv -m gigaam-v3-e2e-rnnt -m t-tech/t-one   # model comparison
stt-ru serve                                                     # http://127.0.0.1:8000
```

```python
>>> from stt_ru import Transcriber

>>> result = Transcriber().transcribe("call.wav")
>>> for segment in result.segments:
...     print(segment.speaker, segment.start, segment.end, segment.text)
Канал 1 0.994 9.054 Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии.
Канал 0 3.49 3.934 Угу
Канал 0 7.682 8.222 Угу
Канал 1 9.634 12.638 Я в свою очередь с вами прощаюсь. Всего доброго, до свидания.
Канал 0 12.994 13.662 До свидания.

>>> result.save("call.srt")
```

The command messages and speaker names ("Канал N", Channel N) are in Russian.

## Docker

```bash
docker run -d -p 8000:8000 -e STT_RU_TOKEN=secret ghcr.io/sergeyshk/stt-ru           # web service
docker run --rm -v "$PWD:/data" ghcr.io/sergeyshk/stt-ru transcribe call.wav         # one file
docker run -d --gpus all -p 8000:8000 -e STT_RU_TOKEN=secret ghcr.io/sergeyshk/stt-ru:latest-gpu
```

The default model is already in the image: the container recognizes speech without network. Volumes, write permissions, Compose and the GPU image requirements are on the [Docker](https://sergeyshk.github.io/Speech-to-Text-Russian/en/docker/) page.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
git clone https://github.com/SergeyShk/Speech-to-Text-Russian.git
cd Speech-to-Text-Russian

make deps        # create the environment and install dependencies
make test        # run the tests and docstring examples (doctest)
make lint        # ruff + mypy
make docker      # build the Docker image
```

All commands - `make help`. The documentation is bilingual: Russian pages are `docs/*.md`, English ones are `docs/*.en.md` next to them; when editing a page, update both versions.

Before sending changes, install the hooks that run the linters on commit and the tests on push:

```bash
uv run pre-commit install
```

## Contributing

Bug reports, ideas and pull requests are welcome, in Russian or English - [issues](https://github.com/SergeyShk/Speech-to-Text-Russian/issues) are open. How the process works and what to check before a pull request is in [CONTRIBUTING.md](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/CONTRIBUTING.md); the rules of conduct are in the [code of conduct](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/CODE_OF_CONDUCT.md).

<details>
<summary><b>Project structure</b></summary>

<br>

*   **docs** - MkDocs documentation in Russian and English
*   **docker** - Docker images for CPU and NVIDIA GPU, Docker Compose files
*   **src/stt_ru**:
    *   pipeline.py - transcription of files and in-memory recordings
    *   result.py - transcription result: segments and recording information
    *   batch.py - batch transcription of files and folder monitoring
    *   cli.py - the `stt-ru` command
    *   exceptions.py - library exceptions
    *   **audio** - reading audio through soundfile and ffmpeg
    *   **asr** - recognition with an onnx-asr model
    *   **vad** - Silero VAD speech detection parameters
    *   **export** - output formats
    *   **evaluation** - quality evaluation on references
    *   **service** - web interface, REST API and job queue
*   **tests** - tests mirroring the package structure

</details>

## Authors

*   Sergey Shkarin (kouki.sergey@gmail.com)

## License

The code is distributed under the [MIT](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/LICENSE.txt) license. The GigaAM-v3 model is distributed under the MIT license; other onnx-asr models have their own licenses.
