# Примеры

Ноутбуки с примерами работы stt-ru. Выводы ячеек сохранены, ноутбуки можно читать без запуска; кнопка открывает ноутбук в Google Colab, где он сам ставит stt-ru.

| Ноутбук | О чем | Colab |
|---|---|---|
| [01_quickstart.ipynb](01_quickstart.ipynb) | Стереозапись звонка от файла до результата: фрагменты по говорящим, форма волны, форматы, сведение каналов, команда `stt-ru transcribe` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/01_quickstart.ipynb) |
| [02_call_analytics.ipynb](02_call_analytics.ipynb) | Аналитика звонка: время речи и тишины, темп, реакция, поддакивания и перебивания, поиск фраз, сводная таблица по папке звонков | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/02_call_analytics.ipynb) |
| [03_long_recording.ipynb](03_long_recording.ipynb) | Глава аудиокниги на 17 минут: скорость распознавания, субтитры SRT и VTT, параметры поиска речи под субтитры | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/03_long_recording.ipynb) |
| [04_model_evaluation.ipynb](04_model_evaluation.ipynb) | `stt-ru evaluate` на фразах FLEURS: WER, CER и скорость моделей, числа цифрами и словами, разбор ошибок, веса int8 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/04_model_evaluation.ipynb) |
| [05_web_service.ipynb](05_web_service.ipynb) | `stt-ru serve` и REST API из Python: очередь, результат в форматах, форма волны, токен, ошибки, клиент для пакета файлов | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergeyShk/Speech-to-Text-Russian/blob/master/notebooks/05_web_service.ipynb) |

Локально: `uv sync --all-extras --group notebooks`, затем `uv run jupyter lab notebooks/`. `make notebooks` выполняет все ноутбуки заново, записывает выводы в файлы и убирает метаданные выполнения. Команды `stt-ru` в ноутбуках выполняются в ячейках `%%bash`: в Linux, macOS и Colab bash есть, в Windows нужен bash из [Git for Windows](https://gitforwindows.org/) в `PATH`.
