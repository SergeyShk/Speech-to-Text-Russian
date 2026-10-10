# stt-ru

![stt-ru](img/logo.jpg)

**stt-ru** is a tool for Russian speech recognition: a Python library, the `stt-ru` command, a web service and a Docker image. Speech is recognized by the [GigaAM-v3](https://github.com/salute-developers/GigaAM) model through [onnx-asr](https://github.com/istupakov/onnx-asr): the text comes with punctuation and digits, a stereo recording is split by channels, and the result is saved as subtitles and tables.

!!! note "Work in progress"
    The first release of the package is 0.3.0. Until then, the working Kaldi version is described in the [README](https://github.com/SergeyShk/Speech-to-Text-Russian#readme).
