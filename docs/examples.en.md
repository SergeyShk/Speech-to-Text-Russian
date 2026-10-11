# Examples

Jupyter notebooks in the [notebooks/](https://github.com/SergeyShk/Speech-to-Text-Russian/tree/master/notebooks) directory of the repository. Each opens in Google Colab with a button and installs stt-ru itself; the outputs of all cells are stored in the files, so the notebooks can be read without running, and the recordings can be played in the embedded players. The notebooks download their recordings and references themselves. The notebooks are in Russian.

## Quick start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/01_quickstart.ipynb)

[01_quickstart.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/01_quickstart.ipynb) - a stereo phone call from reading the file to the result: `load_audio` and `Transcriber`, segments by speaker in a table, a waveform from `Audio.peaks` with the detected speech segments, every output format via `save` and `render`, mixing the channels down (`split_channels=False`), transcribing a single channel from memory via `transcribe_audio`, and the same work with the `stt-ru transcribe` command.

## Call analytics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/02_call_analytics.ipynb)

[02_call_analytics.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/02_call_analytics.ipynb) - what the transcription of a stereo recording tells about a conversation: a timeline of turns by role, speech and silence time, pace in words per minute, response time, backchannels and interruptions, searching for key phrases in the normalized text with regular expressions, transcribing a folder of calls with the command and a `--summary` file, and a summary table by call.

## Long recording and subtitles

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/03_long_recording.ipynb)

[03_long_recording.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/03_long_recording.ipynb) - the first chapter of the audiobook "Dubrovsky" (17 minutes, MP3, LibriVox): reading through ffmpeg, reading and recognition time separately, speed relative to real time, SRT and VTT subtitles, the distribution of segment durations, and tuning the speech detection parameters `min_silence_duration_ms` and `max_speech_duration_s` for subtitles.

## Evaluating models on your own references

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/04_model_evaluation.ipynb)

[04_model_evaluation.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/04_model_evaluation.ipynb) - how to choose a model for your recordings with the `stt-ru evaluate` command: a CSV manifest of 100 [FLEURS](https://huggingface.co/datasets/google/fleurs) phrases, WER, CER and speed of four models, the detailed JSON report, the effect of numbers written as digits or words, text normalization, the recordings with the most errors, int8 against original weights from Python, and the format of your own manifest. The notebook shows how to use the evaluation; choose a model on your own recordings.

## Web service and REST API from Python

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/05_web_service.ipynb)

[05_web_service.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/05_web_service.ipynb) - `stt-ru serve` in the background with an access token and working with it through `httpx`: queueing a file and waiting for the result, the result in different formats, the waveform and the recording with seeking, error responses, deleting a job, a client for a batch of files, and running the same service in Docker.

## Running locally

``` bash
uv sync --all-extras --group notebooks
uv run jupyter lab notebooks/
```

The `stt-ru` commands in the notebooks run in `%%bash` cells: bash is available on Linux, macOS and Colab; on Windows the bash from [Git for Windows](https://gitforwindows.org/) must be in `PATH`.

`make notebooks` runs all notebooks, writes the outputs to the files and removes the execution metadata; the `notebooks.yml` workflow runs the notebooks on a schedule and on demand.
