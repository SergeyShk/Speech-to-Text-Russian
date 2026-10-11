# Модели

## Модель по умолчанию

Речь распознает [GigaAM-v3](https://github.com/salute-developers/GigaAM) E2E RNN-T (`gigaam-v3-e2e-rnnt`) - открытая модель распознавания русской речи от SberDevices под лицензией MIT. Она сразу пишет текст с пунктуацией, заглавными буквами и числами цифрами. Модель работает через [onnx-asr](https://github.com/istupakov/onnx-asr) и ONNX Runtime, без PyTorch, по умолчанию - с весами int8.

## Другие модели

Любую модель onnx-asr можно выбрать по имени параметром `-m` или аргументом `model` у `Transcriber`; при первом запуске она скачивается с Hugging Face. Модели, которые распознают русский:

| Модель | Особенности |
| :----- | :---------- |
| `gigaam-v3-e2e-rnnt`, `gigaam-v3-e2e-ctc` | Пунктуация, заглавные буквы и числа цифрами; CTC обычно быстрее RNN-T |
| `gigaam-v3-rnnt`, `gigaam-v3-ctc` | Строчные буквы без пунктуации, числа словами |
| `gigaam-v2-rnnt`, `gigaam-v2-ctc` | Предыдущая версия GigaAM |
| `t-tech/t-one` | Модель Т-Банка для телефонной речи |
| `nemo-fastconformer-ru-rnnt`, `nemo-fastconformer-ru-ctc` | NVIDIA NeMo FastConformer для русского с пунктуацией |
| `alphacep/vosk-model-ru`, `alphacep/vosk-model-small-ru` | Модели Vosk; small - для слабых компьютеров |
| `onnx-community/whisper-large-v3-turbo` | Многоязычная модель OpenAI Whisper; заметно тяжелее остальных |
| `nemo-parakeet-tdt-0.6b-v3` | Многоязычная модель NVIDIA |

Полный список и форматы файлов моделей - в [документации onnx-asr](https://github.com/istupakov/onnx-asr). Какая модель лучше для ваших записей, покажет [оценка качества](evaluation.md) на своей разметке:

``` bash
stt-ru evaluate eval.csv -m gigaam-v3-e2e-rnnt -m t-tech/t-one -m alphacep/vosk-model-ru
```

## Параметры { #options }

Параметры модели, устройства и поиска речи общие для команд `transcribe`, `evaluate` и `serve`.

| Параметр | По умолчанию | Описание |
| :------- | :----------- | :------- |
| `-m`, `--model` | `gigaam-v3-e2e-rnnt` | Модель onnx-asr; у `evaluate` можно повторить |
| `--quantization` | `int8` | Квантизация весов: `int8` или `none` - исходные веса |
| `--model-dir` | - | Каталог с файлами модели; у `evaluate` - только для одной модели |
| `--device` | `auto` | Устройство: `auto`, `cpu`, `cuda`, `coreml`, `directml` (см. [Установка](installation.md#devices)) |
| `--mono` | выключен | Сводить каналы в один |
| `--ffmpeg` | - | Путь к программе ffmpeg |
| `--vad-threshold`, `--min-speech-ms`, `--max-speech-s`, `--min-silence-ms`, `--speech-pad-ms` | см. [ниже](#vad) | Параметры поиска речи |

## Квантизация

Веса int8 примерно вчетверо меньше исходных и обычно быстрее на CPU. `--quantization none` берет исходные веса; разницу в качестве на своих записях покажет [оценка качества](evaluation.md) с тем и другим значением. Если у модели нет весов int8, загружаются исходные с предупреждением.

## Файлы модели на диске { #local }

`--model-dir` (аргумент `model_dir`) берет файлы модели распознавания из каталога: например, модель, скачанную вручную. Если каталога нет, модель скачивается в него. Silero VAD при этом берется из кэша Hugging Face, поэтому на компьютере без интернета проще скопировать кэш целиком и указать его в `HF_HOME` (см. [Установка](installation.md#models)).

``` bash
stt-ru transcribe call.wav -m gigaam-v3-e2e-rnnt --model-dir /models/gigaam-v3-e2e-rnnt
```

## Поиск речи { #vad }

!!! info ""
    **stt_ru.vad.VadOptions**

Перед распознаванием запись режется на фрагменты речи: модели принимают фрагменты не длиннее 20-30 с. Речь ищет [Silero VAD](https://github.com/snakers4/silero-vad), значения по умолчанию - его и onnx-asr.

| Параметр команды | Поле `VadOptions` | По умолчанию | Описание |
| :--------------- | :---------------- | :----------- | :------- |
| `--vad-threshold` | `threshold` | `0.5` | Вероятность речи, с которой начинается фрагмент, от 0 до 1 |
| - | `neg_threshold` | `threshold - 0.15` | Вероятность, ниже которой фрагмент заканчивается |
| `--min-speech-ms` | `min_speech_duration_ms` | `250` | Наименьшая длительность фрагмента, мс: короткие отбрасываются |
| `--max-speech-s` | `max_speech_duration_s` | `20` | Наибольшая длительность фрагмента, с: длинные режутся |
| `--min-silence-ms` | `min_silence_duration_ms` | `100` | Наименьшая пауза, которая разделяет фрагменты, мс |
| `--speech-pad-ms` | `speech_pad_ms` | `30` | Запас по краям фрагмента, мс |

*   Шумная запись дает фрагменты из шума - поднимите `--vad-threshold`, например до 0.6.
*   Тихую речь модель пропускает - опустите порог до 0.35-0.4.
*   Реплики делятся на слишком короткие фрагменты - увеличьте `--min-silence-ms`, например до 300: фрагменты станут длиннее, субтитры - реже.
*   `max_speech_duration_s` должна быть хотя бы на секунду больше двух `speech_pad_ms`.

``` python
>>> from stt_ru import Transcriber
>>> from stt_ru.vad import VadOptions

>>> transcriber = Transcriber(vad=VadOptions(threshold=0.6, min_silence_duration_ms=300))
```
