# Speech-to-Text (Russian)

<p align="center">
<img src="docs/img/logo.jpg" width="800">
</p>

**stt-ru** - распознавание русской речи: библиотека Python, команда `stt-ru`, веб-интерфейс с REST API и Docker-образ. Речь распознает модель [GigaAM-v3](https://github.com/salute-developers/GigaAM) через [onnx-asr](https://github.com/istupakov/onnx-asr): текст получается с пунктуацией и числами, каналы стереозаписи распознаются как отдельные говорящие, результат сохраняется в TXT, JSON, JSONL, SRT, VTT, ASS и CSV.

Документация: https://sergeyshk.github.io/Speech-to-Text-Russian/

> Пакет готовится к первому выпуску 0.3.0. Версия 0.2 на Kaldi осталась в истории репозитория: [последнее состояние с ней](https://github.com/SergeyShk/Speech-to-Text-Russian/tree/8db5f77d967216fe36194eca067f6b879301b765).

## Установка

Python 3.11 или новее:

```bash
pip install stt-ru                # библиотека и команда stt-ru
pip install "stt-ru[ffmpeg]"      # M4A, AAC, звук из видео без установленного ffmpeg
pip install "stt-ru[service]"     # веб-интерфейс и REST API
pip install "stt-ru[eval]"        # оценка качества моделей на своей разметке
```

Модель (около 230 МБ) скачивается с Hugging Face при первом запуске.

## Использование

```bash
stt-ru transcribe call.wav                                       # текст в консоль
stt-ru transcribe calls/ -o out/ -f srt,ass,csv --summary out/all.csv
stt-ru transcribe calls/ -o out/ --watch                         # мониторинг папки
stt-ru evaluate eval.csv -m gigaam-v3-e2e-rnnt -m gigaam-v3-rnnt # сравнение моделей
stt-ru serve                                                     # http://127.0.0.1:8000
```

```python
from stt_ru import Transcriber

result = Transcriber().transcribe("call.wav")
for segment in result.segments:
    print(segment.speaker, segment.start, segment.end, segment.text)
result.save("call.srt")
```

Параметры команд - `stt-ru <команда> --help`.

## Docker

Образы публикуются в GHCR начиная с выпуска 0.3.0; модель по умолчанию уже в образе.

```bash
docker run -p 8000:8000 -e STT_RU_TOKEN=секрет ghcr.io/sergeyshk/stt-ru     # сервис
docker run --rm -v "$PWD:/data" ghcr.io/sergeyshk/stt-ru transcribe call.wav  # один файл
docker compose -f docker/compose.yaml up -d                                   # сервис через Compose
```

Для NVIDIA GPU - образ `ghcr.io/sergeyshk/stt-ru:latest-gpu` и `docker/compose.gpu.yaml`. Собрать образ из репозитория: `make docker` или `make docker-gpu`.

## Лицензия

Код распространяется под лицензией MIT, см. файл **LICENSE.txt**. Модель GigaAM-v3 распространяется под лицензией MIT.
