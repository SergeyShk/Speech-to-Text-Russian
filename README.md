<p align="center">
  <img src="https://raw.githubusercontent.com/SergeyShk/Speech-to-Text-Russian/master/docs/img/stt-ru.png" alt="stt-ru" width="480">
</p>

<h1 align="center">stt-ru</h1>

<p align="center">
  <b>Speech-to-Text Russian</b> - распознавание русской речи: библиотека Python, команда, веб-сервис и Docker-образ
</p>

<p align="center">
  <a href="https://sergeyshk.github.io/Speech-to-Text-Russian/">Документация</a> ·
  <a href="https://pypi.org/project/stt-ru/">PyPI</a> ·
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/pkgs/container/stt-ru">Docker</a> ·
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/README.en.md">English</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/stt-ru/"><img src="https://img.shields.io/pypi/v/stt-ru?logo=pypi&logoColor=FFE873" alt="Версия"></a>
  <a href="https://pypi.org/project/stt-ru/"><img src="https://img.shields.io/pypi/pyversions/stt-ru.svg?logo=python&logoColor=FFE873" alt="Поддерживаемые версии Python"></a>
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/actions/workflows/ci.yml"><img src="https://github.com/SergeyShk/Speech-to-Text-Russian/actions/workflows/ci.yml/badge.svg" alt="Сборка"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/sergeyshk/speech-to-text-russian.svg" alt="Лицензия"></a>
  <a href="https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/01_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

---

**stt-ru** распознает русскую речь в записях разговоров, лекций и диктовки. Речь распознает модель [GigaAM-v3](https://github.com/salute-developers/GigaAM) через [onnx-asr](https://github.com/istupakov/onnx-asr): текст получается с пунктуацией и числами, каналы стереозаписи распознаются как отдельные говорящие, результат сохраняется в субтитры и таблицы. Модель работает на CPU без PyTorch, на GPU NVIDIA - через ONNX Runtime.

* **[Команда transcribe](https://sergeyshk.github.io/Speech-to-Text-Russian/transcribe/)** - файлы и каталоги, результаты каждого файла и общий файл на пакет, мониторинг папки с новыми записями
* **[Форматы результата](https://sergeyshk.github.io/Speech-to-Text-Russian/formats/)** - TXT, JSON, JSONL, SRT, VTT, ASS для Aegisub и CSV для таблиц
* **[Библиотека Python](https://sergeyshk.github.io/Speech-to-Text-Russian/python/)** - распознавание файлов и записей в памяти, экспорт результата
* **[Модели](https://sergeyshk.github.io/Speech-to-Text-Russian/models/)** - любая модель onnx-asr: GigaAM, T-one, Vosk, Whisper и другие; устройства и параметры поиска речи
* **[Оценка качества](https://sergeyshk.github.io/Speech-to-Text-Russian/evaluation/)** - WER, CER и скорость моделей на своей разметке
* **[Веб-сервис](https://sergeyshk.github.io/Speech-to-Text-Russian/service/)** - веб-интерфейс с формой волны и переходом по фрагментам, REST API, очередь и токен доступа
* **[Docker](https://sergeyshk.github.io/Speech-to-Text-Russian/docker/)** - образы для CPU и GPU NVIDIA с моделью внутри

## Установка

Требуется Python 3.11 или новее.

```bash
pip install stt-ru
```

Или с помощью [uv](https://docs.astral.sh/uv/):

```bash
uv add stt-ru
```

Дополнения: `ffmpeg` - форматы M4A, AAC, звук из видео без установленного ffmpeg; `service` - веб-интерфейс и REST API; `eval` - оценка качества. Все сразу:

```bash
pip install "stt-ru[ffmpeg,service,eval]"
```

Модель (около 230 МБ) скачивается с Hugging Face при первом запуске. GPU и работа без интернета - на странице [Установка](https://sergeyshk.github.io/Speech-to-Text-Russian/installation/).

## Быстрый старт

```bash
stt-ru transcribe call.wav                                       # текст в консоль
stt-ru transcribe calls/ -o out/ -f srt,ass,csv --summary out/all.csv
stt-ru transcribe calls/ -o out/ --watch                         # мониторинг папки
stt-ru evaluate eval.csv -m gigaam-v3-e2e-rnnt -m t-tech/t-one   # сравнение моделей
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
PosixPath('call.srt')
```

Ноутбуки с примерами - в каталоге [notebooks/](https://github.com/SergeyShk/Speech-to-Text-Russian/tree/master/notebooks): быстрый старт, аналитика звонка, длинная запись и субтитры, оценка моделей, веб-сервис из Python. Каждый открывается в Colab, описания - на странице [Примеры](https://sergeyshk.github.io/Speech-to-Text-Russian/examples/).

## Docker

```bash
docker run -d -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru          # веб-сервис
docker run --rm -v "$PWD:/data" ghcr.io/sergeyshk/stt-ru transcribe call.wav         # один файл
docker run -d --gpus all -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru:latest-gpu
```

Модель по умолчанию уже в образе: контейнер распознает без сети. Тома, права на запись, Compose и требования GPU-образа - на странице [Docker](https://sergeyshk.github.io/Speech-to-Text-Russian/docker/).

## Разработка

Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями и [ruff](https://docs.astral.sh/ruff/) для линтинга и форматирования.

```bash
git clone https://github.com/SergeyShk/Speech-to-Text-Russian.git
cd Speech-to-Text-Russian

make deps        # создать окружение и установить зависимости
make test        # запустить тесты и примеры из докстрингов (doctest)
make lint        # ruff + mypy
make docker      # собрать Docker-образ
```

Полный список команд - `make help`. Документация двуязычная: русские страницы `docs/*.md`, английские - `docs/*.en.md` рядом с ними; при правке страницы обновляйте обе версии.

Перед отправкой изменений стоит установить хуки, которые прогонят линтеры на коммите и тесты на пуше:

```bash
uv run pre-commit install
```

## Участие в проекте

Баг-репорты, идеи и пул-реквесты приветствуются - [issues](https://github.com/SergeyShk/Speech-to-Text-Russian/issues) открыты. Как устроен процесс и что проверить перед отправкой пул-реквеста - в [CONTRIBUTING.md](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/CONTRIBUTING.md); правила общения - в [кодексе поведения](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/CODE_OF_CONDUCT.md).

<details>
<summary><b>Структура проекта</b></summary>

<br>

*   **docs** - документация MkDocs на русском и английском
*   **docker** - Docker-образы для CPU и GPU NVIDIA, файлы Docker Compose
*   **src/stt_ru**:
    *   pipeline.py - распознавание файлов и записей в памяти
    *   result.py - результат распознавания: фрагменты и сведения о записи
    *   batch.py - пакетное распознавание файлов и мониторинг каталогов
    *   cli.py - команда `stt-ru`
    *   exceptions.py - исключения библиотеки
    *   **audio** - чтение аудио через soundfile и ffmpeg
    *   **asr** - распознавание моделью onnx-asr
    *   **vad** - параметры поиска речи Silero VAD
    *   **export** - форматы результата
    *   **evaluation** - оценка качества на разметке
    *   **service** - веб-интерфейс, REST API и очередь задач
*   **tests** - тесты, повторяющие структуру пакета

</details>

## Авторы

*   Шкарин Сергей (kouki.sergey@gmail.com)

## Лицензия

Код распространяется под лицензией [MIT](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/LICENSE.txt). Модель GigaAM-v3 распространяется под лицензией MIT; у других моделей onnx-asr - свои лицензии.
