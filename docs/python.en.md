# Python library

## Transcription { #transcriber }

!!! info ""
    **stt_ru.Transcriber**

`Transcriber` loads a model once and uses it to transcribe files and in-memory recordings. The channels of a stereo recording are recognized separately, and recordings longer than 10 minutes in windows of about 10 minutes cut at a pause. One object can be used from several threads.

| Parameter | Type | Default | Description |
| :-------- | :--- | :------ | :---------- |
| `model` | str | `"gigaam-v3-e2e-rnnt"` | onnx-asr [model](models.md) name |
| `quantization` | str | `"int8"` | Weight quantization; `None` - original weights |
| `model_dir` | str | `None` | Folder with model files; `None` - the Hugging Face cache |
| `providers` | list[str] | `None` | ONNX Runtime providers by priority; `None` - the available ones except CoreML and TensorRT |
| `vad` | VadOptions | `None` | [Speech detection](models.md#vad) parameters |
| `split_channels` | bool | `True` | Recognize channels separately; `False` - mix them into one |
| `ffmpeg` | str | `None` | Path to ffmpeg; `None` - search in `PATH` and `imageio-ffmpeg` |

| Method | Description |
| :----- | :---------- |
| `transcribe(path)` | Transcribes a file and returns a `Transcript` |
| `transcribe_audio(audio, source=None)` | Transcribes an in-memory recording (`Audio`) |
| `load(path, *, max_duration=None)` | Reads a file with a sample rate the model accepts; `max_duration` in seconds limits the recording length |

``` python
>>> from stt_ru import Transcriber

>>> transcriber = Transcriber()
>>> result = transcriber.transcribe("call.wav")
>>> result.duration, result.channels, len(result.segments)
(15.751, 2, 5)
>>> result.segments[0]
Segment(start=0.994, end=9.054, text='Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии.', channel=1)
```

## Result { #transcript }

!!! info ""
    **stt_ru.Transcript**, **stt_ru.Segment**

`Transcript` is the result of transcribing a recording: segments ordered by start time and information about the recording.

| Attribute | Type | Description |
| :-------- | :--- | :---------- |
| `segments` | tuple[Segment, ...] | Speech segments ordered by start time |
| `duration` | float | Recording duration, s |
| `channels` | int | Number of recognized channels: 1 if the channels are mixed or identical |
| `model` | str | Recognition model |
| `source` | str | Recording file name; `None` - the recording is not from a file |
| `text` | str | Text of all segments joined with spaces |

`save(path, fmt=None)` saves the result in the [format](formats.md) given by the file extension or `fmt`.

`Segment` is a speech segment: `start` and `end` in seconds, `text`, `channel` (channel number from zero; `None` - channels are mixed), `speaker` ("Канал N" or `None`) and `duration`.

## Export { #export }

!!! info ""
    **stt_ru.export**

| Function | Description |
| :------- | :---------- |
| `render(transcript, fmt)` | Result text in a format from `FORMATS` |
| `dumps(transcript, fmt)` | File content as bytes: UTF-8, CSV with a BOM |
| `save(transcript, path, fmt=None)` | Writes a file entirely; the format comes from the extension or `fmt` |
| `to_txt`, `to_json`, `to_jsonl`, `to_srt`, `to_vtt`, `to_ass`, `to_csv` | Text in one format |

`FORMATS` is the tuple of supported formats: `("txt", "json", "jsonl", "srt", "vtt", "ass", "csv")`.

``` python
>>> from stt_ru.export import render
>>> print("\n".join(render(result, "srt").splitlines()[4:7]))
2
00:00:03,490 --> 00:00:03,934
Канал 0: Угу
```

## Audio { #audio }

!!! info ""
    **stt_ru.audio**

`load_audio(path, *, sample_rate=None, ffmpeg=None, max_duration=None)` reads a file into `Audio`: float32 samples of shape (number of samples, number of channels) and the sample rate. `sample_rate` resamples with ffmpeg; `max_duration` rejects a recording longer than the given number of seconds without decoding it entirely. `Audio` has `num_channels`, `duration`, `channel(index)`, `mono()` and `peaks(points)` - peaks for drawing a waveform. `find_ffmpeg()` returns the path to the ffmpeg that will be used, or `None`.

A recording from another source can be transcribed without a file:

``` python
>>> import numpy as np
>>> from stt_ru.audio import Audio

>>> samples = np.zeros((16000 * 5, 1), dtype=np.float32)    # 5 s of silence
>>> transcriber.transcribe_audio(Audio(samples, 16000)).segments
()
```

## Exceptions { #exceptions }

!!! info ""
    **stt_ru.exceptions**

All exceptions inherit `SttRuError`, and some also inherit a built-in Python class, so they can be caught by either name.

| Exception | Built-in class | When raised |
| :-------- | :------------- | :---------- |
| `SttRuError` | `Exception` | Base class |
| `AudioError` | - | The file cannot be read, it needs ffmpeg and there is none, the recording is longer than `max_duration`, the model does not accept the sample rate |
| `ModelError` | - | The model is not supported, could not be downloaded or its files are corrupted |
| `ParameterError` | `ValueError` | Invalid parameter value: format, speech detection, evaluation references |
| `MissingDependencyError` | `ImportError` | A package from an extra is missing, for example `jiwer` for quality evaluation |

A missing file raises the built-in `FileNotFoundError`.
