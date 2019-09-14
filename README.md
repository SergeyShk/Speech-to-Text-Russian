# Speech-to-Text (Russian)

<p align="center">
<img src="https://www.analyticsindiamag.com/wp-content/uploads/2019/02/1_ChocH_eUxil5eaeXIsd3rw.png" width="800">
</p>

Проект для распознавания речи на русском языке на основе pykaldi.

## Установка
### Самостоятельная (Linux)

1. Установить kaldi

https://kaldi-asr.org/doc/tutorial_setup.html

2. Установить необходимые Python-библиотеки:

`$ pip install -r requirements.txt`

3. Добавить в PATH пути к компонентам kaldi:

`$ PATH /kaldi/src/featbin:/kaldi/src/ivectorbin:/kaldi/src/online2bin:/kaldi/src/rnnlmbin:/kaldi/src/fstbin:$PATH`

4. Склонировать репозиторий проекта:

`$ git clone https://github.com/SergeyShk/Speech-to-Text-Russian.git`

### Docker

1. Собрать docker-образ:

`$ docker build -t speech_recognition:latest .`

2. Создать docker-том для работы с внешними данными:

`$ docker volume create -d local -o o=bind -o device=[DIR] asr_volume`

3. Запустить docker-контейнер:

`$ docker run -it --rm -p 9000:9000 -p 5000:5000 -v asr_volume:/archive speech_recognition`

## Структура проекта

Файлы проекта расположены в директории /speech_recognition:

* **start_recognition.py** - скрипт запуска процедуры распознавания;
* **/tools** - набор инструментов для распознавания:
    * **data_preparator.py** - скрипт подготовки данных для распознавания;
    * **recognizer.py** - скрипт распознавания речи;
    * **segmenter.py** - скрипт сегментации речи;
    * **transcriptins_parser.py** - скрипт парсинга результатов распознавания;
* **/model** - набор файлов для модели распознавания.

## Модель

В качестве акустической и языковой модели используется русскоязычная модель от alphacep:

http://alphacephei.com/kaldi/kaldi-ru-0.6.tar.gz

При необходимости использования собственной модели, необходимо заменить соответствующие файлы в директории /model.

## Запуск
### Распознавание речи

1. Подготовить директорию для размещения WAV-файлов;
2. Для запуска процедуры распознавания речи выполнить команду:

`$ ./start_recognition.py /archive/wav /archive/output -dw -l`

3. Для запуска режима мониторинга директории выполнить команду:

`$ ./start_recognition.py /archive/wav /archive/output -l -t 60 -d 1`

Описание параметров запуска доступно по команде:

`$ ./start_recognition.py -h`

```console
usage: start_recognition.py [-h] [-rm REC_MODEL] [-rg REC_GRAPH]
                            [-rw REC_WORDS] [-rc REC_CONF] [-ri REC_ICONF]
                            [-sm SEGM_MODEL] [-sc SEGM_CONF] [-sp SEGM_POST]
                            [-p PROCESSES] [-l] [-dw] [-t TIME] [-d DELTA]
                            WAV OUT

Запуск процедуры распознавания речи

positional arguments:
  WAV                   Путь к .WAV файлам аудио
  OUT                   Путь к директории с результатами распознавания

optional arguments:
  -h, --help            show this help message and exit
  -rm REC_MODEL, --rec_model REC_MODEL
                        Путь к .MDL файлу модели распознавания
  -rg REC_GRAPH, --rec_graph REC_GRAPH
                        Путь к .FST файлу общего графа распознавания
  -rw REC_WORDS, --rec_words REC_WORDS
                        Путь к .TXT файлу текстового корпуса
  -rc REC_CONF, --rec_conf REC_CONF
                        Путь к .CONF конфигурационному файлу распознавания
  -ri REC_ICONF, --rec_iconf REC_ICONF
                        Путь к .CONF конфигурационному файлу векторного
                        экстрактора
  -sm SEGM_MODEL, --segm_model SEGM_MODEL
                        Путь к .RAW файлу модели сегментации
  -sc SEGM_CONF, --segm_conf SEGM_CONF
                        Путь к .CONF конфигурационному файлу сегментации
  -sp SEGM_POST, --segm_post SEGM_POST
                        Путь к .VEC файлу апостериорных вероятностей
                        сегментации
  -p PROCESSES, --processes PROCESSES
                        Количество процессов для обработки файлов
  -l, --log             Логировать результат распознавания
  -dw, --delete_wav     Удалять .WAV файлы после распознавания
  -t TIME, --time TIME  Пауза перед очередным сканированием директории в
                        секундах
  -d DELTA, --delta DELTA
                        Дельта, выдерживаемая до чтения файла в минутах
```

### Сервис ноутбуков

1. Запустить сервис:

`$ jupyter notebook --no-browser --ip=0.0.0.0 --port=9000 --allow-root`

2. Перейти по адресу:

`http://0.0.0.0:9000`
