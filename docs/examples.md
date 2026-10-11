# Примеры

Ноутбуки Jupyter в каталоге [notebooks/](https://github.com/SergeyShk/Speech-to-Text-Russian/tree/master/notebooks) репозитория. Каждый открывается в Google Colab кнопкой и сам ставит stt-ru; выводы всех ячеек сохранены в файлах, поэтому ноутбуки можно читать и без запуска. Записи и разметку ноутбуки скачивают сами.

## Быстрый старт

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/01_quickstart.ipynb)

[01_quickstart.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/01_quickstart.ipynb) - стереозапись телефонного разговора от чтения файла до результата: `load_audio` и `Transcriber`, фрагменты по говорящим в таблице, форма волны по `Audio.peaks` с найденными фрагментами речи, все форматы результата через `save` и `render`, сведение каналов (`split_channels=False`), распознавание одного канала из памяти через `transcribe_audio` и та же работа командой `stt-ru transcribe`.

## Аналитика звонка

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/02_call_analytics.ipynb)

[02_call_analytics.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/02_call_analytics.ipynb) - что можно узнать о разговоре по результату распознавания стереозаписи: лента реплик по ролям, время речи и тишины, темп в словах в минуту, время реакции, поддакивания и перебивания, поиск ключевых фраз по нормализованному тексту регулярными выражениями, распознавание папки звонков командой с общим файлом `--summary` и сводная таблица по звонкам.

## Длинная запись и субтитры

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/03_long_recording.ipynb)

[03_long_recording.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/03_long_recording.ipynb) - первая глава аудиокниги «Дубровский» (17 минут, MP3, LibriVox): чтение через ffmpeg, время чтения и распознавания отдельно, скорость относительно реального времени, субтитры SRT и VTT, распределение длительности фрагментов и подбор параметров поиска речи `min_silence_duration_ms` и `max_speech_duration_s` под субтитры.

## Оценка моделей на своей разметке

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/04_model_evaluation.ipynb)

[04_model_evaluation.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/04_model_evaluation.ipynb) - как выбрать модель для своих записей командой `stt-ru evaluate`: разметка CSV из 100 фраз [FLEURS](https://huggingface.co/datasets/google/fleurs), WER, CER и скорость четырех моделей, подробный отчет JSON, влияние записи чисел цифрами и словами, нормализация текста, записи с наибольшими ошибками, веса int8 против исходных из Python и формат своей разметки. Ноутбук показывает, как пользоваться оценкой; выбирать модель стоит по своим записям.

## Веб-сервис и REST API из Python

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/05_web_service.ipynb)

[05_web_service.ipynb](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/05_web_service.ipynb) - `stt-ru serve` в фоне с токеном доступа и работа с ним через `httpx`: постановка файла в очередь и ожидание результата, результат в разных форматах, форма волны и запись с перемоткой, ответы с ошибками, удаление задачи, клиент для пакета файлов и запуск того же сервиса в Docker.

## Запуск локально

``` bash
uv sync --all-extras --group notebooks
uv run jupyter lab notebooks/
```

Команды `stt-ru` в ноутбуках выполняются в ячейках `%%bash`: в Linux, macOS и Colab bash есть, в Windows нужен bash из [Git for Windows](https://gitforwindows.org/) в `PATH`.

Команда `make notebooks` выполняет все ноутбуки, записывает выводы в файлы и убирает метаданные выполнения; workflow `notebooks.yml` прогоняет ноутбуки по расписанию и вручную.
