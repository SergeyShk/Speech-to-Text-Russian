# stt-ru

![stt-ru](img/stt-ru.svg)

**stt-ru** is a tool for Russian speech recognition: a Python library, the `stt-ru` command, a web interface with a REST API and a Docker image. Speech is recognized by the [GigaAM-v3](https://github.com/salute-developers/GigaAM) model through [onnx-asr](https://github.com/istupakov/onnx-asr): the text comes with punctuation and digits, the channels of a stereo recording are recognized as separate speakers, and the result is saved as subtitles and tables.

## Features

*   transcribe [files and folders](transcribe.md) from the command line: text to the console or the results of every file to a folder, one summary file for a whole batch, monitoring a folder for new recordings
*   save the result in the [formats](formats.md) TXT, JSON, JSONL, SRT, VTT, ASS and CSV
*   transcribe recordings from files and from memory in a [Python program](python.md)
*   choose the onnx-asr [model](models.md), the device and the speech detection parameters
*   [evaluate the quality](evaluation.md) of models on your own references: word (WER) and character (CER) error rates, speed
*   run a [web interface and REST API](service.md) for yourself or a team
*   run all of it in [Docker](docker.md), including NVIDIA GPUs

## Installation

Python 3.11 or newer is required.

``` bash
pip install stt-ru
```

The extras for formats via ffmpeg, the web service and quality evaluation, GPUs and the model cache are described on the [Installation](installation.md) page.

## Quick start

``` bash
stt-ru transcribe call.wav                     # text to the console
stt-ru transcribe calls/ -o out/ -f srt,csv    # results of every file in out/
stt-ru serve                                   # web interface at http://127.0.0.1:8000
```

``` python
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
PosixPath('call.srt')
```

On the first run the model (about 230 MB) is downloaded from Hugging Face.

## How it works

1.  The file is read entirely: WAV, FLAC, OGG, MP3 and other uncompressed formats by the soundfile library, M4A, AAC, AMR, WMA and the sound of videos by ffmpeg.
2.  The channels of a stereo recording are recognized separately: in a recorded conversation each channel is its own speaker ("Канал 0", "Канал 1", that is "Channel 0", "Channel 1"). Identical channels are recognized once.
3.  [Silero VAD](models.md#vad) speech detection cuts the recording into speech segments of at most 20 s, and the model recognizes each segment.
4.  The result is a list of segments with start, end, channel and text ordered by start time. Recordings longer than 10 minutes are recognized in windows of about 10 minutes cut at a pause, so recognition needs less memory.

!!! note "Version 0.2"
    The previous Kaldi version is no longer developed. Its code remains in the [repository history](https://github.com/SergeyShk/Speech-to-Text-Russian/tree/8db5f77d967216fe36194eca067f6b879301b765).
