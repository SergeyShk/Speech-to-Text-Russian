# Библиотека Python

## Распознавание { #transcriber }

!!! info ""
    **stt_ru.Transcriber**

`Transcriber` загружает модель один раз и распознает ею файлы и записи в памяти. Каналы стереозаписи распознаются отдельно, записи длиннее 10 минут - окнами около 10 минут с границей в паузе. Один объект можно использовать из нескольких потоков.

| Параметр | Тип | По умолчанию | Описание |
| :------- | :-- | :----------- | :------- |
| `model` | str | `"gigaam-v3-e2e-rnnt"` | Имя [модели](models.md) onnx-asr |
| `quantization` | str | `"int8"` | Квантизация весов; `None` - исходные веса |
| `model_dir` | str | `None` | Каталог с файлами модели; `None` - кэш Hugging Face |
| `providers` | list[str] | `None` | Провайдеры ONNX Runtime по приоритету; `None` - доступные, кроме CoreML и TensorRT |
| `vad` | VadOptions | `None` | Параметры [поиска речи](models.md#vad) |
| `split_channels` | bool | `True` | Распознавать каналы отдельно; `False` - сводить в один |
| `ffmpeg` | str | `None` | Путь к ffmpeg; `None` - поиск в `PATH` и в `imageio-ffmpeg` |

| Метод | Описание |
| :---- | :------- |
| `transcribe(path)` | Распознает файл и возвращает `Transcript` |
| `transcribe_audio(audio, source=None)` | Распознает запись в памяти (`Audio`) |
| `load(path, *, max_duration=None)` | Читает файл с частотой, которую принимает модель; `max_duration` в секундах ограничивает длину записи |

``` python
>>> from stt_ru import Transcriber

>>> transcriber = Transcriber()
>>> result = transcriber.transcribe("call.wav")
>>> result.duration, result.channels, len(result.segments)
(15.751, 2, 5)
>>> result.segments[0]
Segment(start=0.994, end=9.054, text='Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии.', channel=1)
```

## Результат { #transcript }

!!! info ""
    **stt_ru.Transcript**, **stt_ru.Segment**

`Transcript` - результат распознавания записи: фрагменты по времени начала и сведения о записи.

| Атрибут | Тип | Описание |
| :------ | :-- | :------- |
| `segments` | tuple[Segment, ...] | Фрагменты речи по времени начала |
| `duration` | float | Длительность записи, с |
| `channels` | int | Число распознанных каналов: 1, если каналы сведены или одинаковы |
| `model` | str | Модель распознавания |
| `source` | str | Имя файла записи; `None` - запись не из файла |
| `channel_map` | tuple[int \| None, ...] | Для каждого канала записи - номер канала, под которым распознана его речь: свой, совпавшего с ним или `None`, если дорожка одна |
| `text` | str | Текст всех фрагментов через пробел |

`save(path, fmt=None)` сохраняет результат в [формате](formats.md) по расширению файла или из `fmt`.

`Segment` - фрагмент речи: `start` и `end` в секундах, `text`, `channel` (номер канала с нуля; `None` - каналы сведены), `speaker` («Канал N» или `None`) и `duration`.

## Экспорт { #export }

!!! info ""
    **stt_ru.export**

| Функция | Описание |
| :------ | :------- |
| `render(transcript, fmt)` | Текст результата в формате из `FORMATS` |
| `dumps(transcript, fmt)` | Содержимое файла в байтах: UTF-8, у CSV - с BOM |
| `save(transcript, path, fmt=None)` | Запись в файл целиком; формат - по расширению или из `fmt` |
| `to_txt`, `to_json`, `to_jsonl`, `to_srt`, `to_vtt`, `to_ass`, `to_csv` | Текст в одном формате |

`FORMATS` - кортеж поддерживаемых форматов: `("txt", "json", "jsonl", "srt", "vtt", "ass", "csv")`.

``` python
>>> from stt_ru.export import render
>>> print("\n".join(render(result, "srt").splitlines()[4:7]))
2
00:00:03,490 --> 00:00:03,934
Канал 0: Угу
```

## Аудио { #audio }

!!! info ""
    **stt_ru.audio**

`load_audio(path, *, sample_rate=None, ffmpeg=None, max_duration=None)` читает файл в `Audio`: отсчеты float32 формы (число отсчетов, число каналов) и частоту. `sample_rate` меняет частоту через ffmpeg, `max_duration` отклоняет запись длиннее заданного числа секунд, не декодируя ее целиком. У `Audio` есть `num_channels`, `duration`, `channel(index)`, `mono()` и `peaks(points)` - пики для рисования формы волны. `find_ffmpeg()` возвращает путь к ffmpeg, который будет использован, или `None`.

Запись из другого источника можно распознать без файла:

``` python
>>> import numpy as np
>>> from stt_ru.audio import Audio

>>> samples = np.zeros((16000 * 5, 1), dtype=np.float32)    # 5 с тишины
>>> transcriber.transcribe_audio(Audio(samples, 16000)).segments
()
```

## Исключения { #exceptions }

!!! info ""
    **stt_ru.exceptions**

Все исключения наследуют `SttRuError`, а некоторые - еще и встроенный класс Python, поэтому их можно ловить по любому из имен.

| Исключение | Встроенный класс | Когда поднимается |
| :--------- | :--------------- | :---------------- |
| `SttRuError` | `Exception` | Базовый класс |
| `AudioError` | - | Файл не читается, для него нужен ffmpeg, а его нет, запись длиннее `max_duration`, модель не принимает частоту |
| `ModelError` | - | Модель не поддерживается, ее не удалось скачать или ее файлы повреждены |
| `ParameterError` | `ValueError` | Недопустимое значение параметра: формат, поиск речи, разметка оценки |
| `MissingDependencyError` | `ImportError` | Нет пакета из дополнения, например `jiwer` для оценки качества |

Отсутствующий файл дает встроенное `FileNotFoundError`.
