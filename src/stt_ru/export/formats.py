import csv
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..exceptions import ParameterError
from ..result import Segment, Transcript

CSV_COLUMNS = ("Audio File", "Start", "End", "Name", "Text")


def _milliseconds(seconds: float) -> int:
    return max(round(seconds * 1000), 0)


def _clock(seconds: float, separator: str) -> str:
    hours, rest = divmod(_milliseconds(seconds), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    whole, milliseconds = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{separator}{milliseconds:03d}"


def _ass_clock(seconds: float) -> str:
    hours, rest = divmod(max(round(seconds * 100), 0), 360_000)
    minutes, rest = divmod(rest, 6000)
    whole, centiseconds = divmod(rest, 100)
    return f"{hours:d}:{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def _line(segment: Segment) -> str:
    return f"{segment.speaker}: {segment.text}" if segment.speaker else segment.text


def _segment(segment: Segment) -> dict[str, Any]:
    return {
        "start": segment.start,
        "end": segment.end,
        "channel": segment.channel,
        "speaker": segment.speaker,
        "text": segment.text,
    }


def to_txt(transcript: Transcript) -> str:
    """
    Текст: фрагмент на строку, у каналов записи - с говорящим

    Аргументы:
        transcript (Transcript): Результат распознавания

    Вывод:
        str: Текст

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import to_txt
        >>> segments = (Segment(0.5, 1.2, "Алло.", 0), Segment(1.0, 3.0, "Здравствуйте!", 1))
        >>> print(to_txt(Transcript(segments, 3.0, 2, "gigaam-v3-e2e-rnnt")))
        Канал 0: Алло.
        Канал 1: Здравствуйте!
        <BLANKLINE>
    """
    return "".join(f"{_line(segment)}\n" for segment in transcript.segments)


def to_json(transcript: Transcript) -> str:
    """
    JSON: сведения о записи и фрагменты

    Аргументы:
        transcript (Transcript): Результат распознавания

    Вывод:
        str: Объект с полями source, model, duration, channels и segments; у фрагмента -
            start и end в секундах, channel, speaker и text

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import to_json
        >>> transcript = Transcript((Segment(0.5, 1.2, "Алло.", 0),), 3.0, 2, "m", "call.wav")
        >>> print(to_json(transcript))
        {
          "source": "call.wav",
          "model": "m",
          "duration": 3.0,
          "channels": 2,
          "segments": [
            {
              "start": 0.5,
              "end": 1.2,
              "channel": 0,
              "speaker": "Канал 0",
              "text": "Алло."
            }
          ]
        }
        <BLANKLINE>
    """
    document = {
        "source": transcript.source,
        "model": transcript.model,
        "duration": transcript.duration,
        "channels": transcript.channels,
        "segments": [_segment(segment) for segment in transcript.segments],
    }
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def to_jsonl(transcript: Transcript) -> str:
    """
    JSON Lines: фрагмент на строку с именем файла записи

    Описание:
        Строки нескольких записей можно дописывать в один файл

    Аргументы:
        transcript (Transcript): Результат распознавания

    Вывод:
        str: Объекты с полями source, start, end, channel, speaker и text по строкам

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import to_jsonl
        >>> transcript = Transcript((Segment(0.5, 1.2, "Алло.", 0),), 3.0, 2, "m", "call.wav")
        >>> print(to_jsonl(transcript), end="")
        {"source": "call.wav", "start": 0.5, "end": 1.2, "channel": 0, "speaker": "Канал 0", "text": "Алло."}
    """
    return "".join(
        json.dumps({"source": transcript.source, **_segment(segment)}, ensure_ascii=False) + "\n"
        for segment in transcript.segments
    )


def to_srt(transcript: Transcript) -> str:
    """
    Субтитры SubRip (SRT)

    Аргументы:
        transcript (Transcript): Результат распознавания

    Вывод:
        str: Субтитры; у каналов записи текст начинается с говорящего

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import to_srt
        >>> print(to_srt(Transcript((Segment(0.994, 9.054, "Алло.", 1),), 10.0, 2, "m")))
        1
        00:00:00,994 --> 00:00:09,054
        Канал 1: Алло.
        <BLANKLINE>
        <BLANKLINE>
    """
    return "".join(
        f"{index}\n{_clock(segment.start, ',')} --> {_clock(segment.end, ',')}\n{_line(segment)}\n\n"
        for index, segment in enumerate(transcript.segments, 1)
    )


def _vtt_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(transcript: Transcript) -> str:
    """
    Субтитры WebVTT

    Аргументы:
        transcript (Transcript): Результат распознавания

    Вывод:
        str: Субтитры; говорящий каналов записи - в теге голоса <v> и перед текстом:
            плееры и ffmpeg показывают только текст

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import to_vtt
        >>> print(to_vtt(Transcript((Segment(0.994, 9.054, "Алло.", 1),), 10.0, 2, "m")))
        WEBVTT
        <BLANKLINE>
        00:00:00.994 --> 00:00:09.054
        <v Канал 1>Канал 1: Алло.
        <BLANKLINE>
        <BLANKLINE>
    """
    cues = []
    for segment in transcript.segments:
        text = _vtt_escape(segment.text)
        if segment.speaker:
            text = f"<v {segment.speaker}>{segment.speaker}: {text}"
        cues.append(f"{_clock(segment.start, '.')} --> {_clock(segment.end, '.')}\n{text}\n\n")
    return "WEBVTT\n\n" + "".join(cues)


_ASS_HEADER = """[Script Info]
; Script generated by stt-ru
WrapStyle: 0
ScaledBorderAndShadow: yes
Collisions: Normal
Title: Default Aegisub file
YCbCr Matrix: None
ScriptType: v4.00+

[Aegisub Project Garbage]
Audio File: {source}
Scroll Position: 0
Active Line: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, \
Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20.0,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100.0,100.0,\
0.0,0.0,1,2.0,2.0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_text(text: str) -> str:
    return text.replace("\n", "\\N")


def to_ass(transcript: Transcript) -> str:
    """
    Субтитры Advanced SubStation Alpha (ASS) для Aegisub

    Описание:
        Формат тот же, что у версии 0.2: имя файла записи - в разделе Aegisub Project Garbage,
        говорящий - в поле Name, время с точностью до сотых секунды

    Аргументы:
        transcript (Transcript): Результат распознавания

    Вывод:
        str: Субтитры

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import to_ass
        >>> transcript = Transcript((Segment(0.994, 9.054, "Алло.", 1),), 10.0, 2, "m", "call.wav")
        >>> print(to_ass(transcript).splitlines()[-1])
        Dialogue: 0,0:00:00.99,0:00:09.05,Default,Канал 1,0,0,0,,Алло.
    """
    events = "".join(
        f"Dialogue: 0,{_ass_clock(segment.start)},{_ass_clock(segment.end)},Default,"
        f"{segment.speaker or ''},0,0,0,,{_ass_text(segment.text)}\n"
        for segment in transcript.segments
    )
    return _ASS_HEADER.format(source=transcript.source or "") + events


def to_csv(transcript: Transcript, *, header: bool = True) -> str:
    """
    Таблица CSV

    Описание:
        Колонки те же, что у версии 0.2: Audio File, Start и End в миллисекундах, Name
        (говорящий) и Text. Строки нескольких записей можно дописывать в один файл
        с header=False

    Аргументы:
        transcript (Transcript): Результат распознавания
        header (bool): Начинать со строки заголовка

    Вывод:
        str: Таблица

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import to_csv
        >>> transcript = Transcript((Segment(0.78, 7.0, "Алло.", 1),), 10.0, 2, "m", "call.wav")
        >>> print(to_csv(transcript), end="")
        Audio File,Start,End,Name,Text
        call.wav,780,7000,Канал 1,Алло.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if header:
        writer.writerow(CSV_COLUMNS)
    for segment in transcript.segments:
        writer.writerow(
            [
                transcript.source or "",
                _milliseconds(segment.start),
                _milliseconds(segment.end),
                segment.speaker or "",
                segment.text,
            ]
        )
    return buffer.getvalue()


_RENDERERS: dict[str, Callable[[Transcript], str]] = {
    "txt": to_txt,
    "json": to_json,
    "jsonl": to_jsonl,
    "srt": to_srt,
    "vtt": to_vtt,
    "ass": to_ass,
    "csv": to_csv,
}
FORMATS = tuple(_RENDERERS)


def _check_format(fmt: str) -> str:
    name = fmt.lower().lstrip(".")
    if name not in _RENDERERS:
        raise ParameterError(f"Формат {fmt} не поддерживается, допустимые: {', '.join(FORMATS)}")
    return name


def render(transcript: Transcript, fmt: str) -> str:
    """
    Результат распознавания в заданном формате

    Аргументы:
        transcript (Transcript): Результат распознавания
        fmt (str): Формат из FORMATS: txt, json, jsonl, srt, vtt, ass, csv

    Вывод:
        str: Текст в формате

    Исключения:
        ParameterError: Если формат не поддерживается
    """
    return _RENDERERS[_check_format(fmt)](transcript)


def dumps(transcript: Transcript, fmt: str) -> bytes:
    """
    Содержимое файла с результатом распознавания

    Описание:
        UTF-8 с переводами строк \\n; CSV - с BOM, чтобы Excel открывал кириллицу

    Аргументы:
        transcript (Transcript): Результат распознавания
        fmt (str): Формат из FORMATS

    Вывод:
        bytes: Содержимое файла

    Исключения:
        ParameterError: Если формат не поддерживается

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> from stt_ru.export import dumps
        >>> transcript = Transcript((Segment(0.78, 7.0, "Алло.", 1),), 10.0, 2, "m", "call.wav")
        >>> dumps(transcript, "csv")[:13]
        b'\\xef\\xbb\\xbfAudio File'
    """
    name = _check_format(fmt)
    content = render(transcript, name).encode("utf-8")
    return b"\xef\xbb\xbf" + content if name == "csv" else content


def save(transcript: Transcript, path: str | os.PathLike[str], fmt: str | None = None) -> Path:
    """
    Сохранение результата распознавания в файл

    Описание:
        Содержимое - как у dumps: UTF-8 с переводами строк \\n, CSV - с BOM

    Аргументы:
        transcript (Transcript): Результат распознавания
        path (str): Путь к файлу
        fmt (str): Формат из FORMATS; None - по расширению файла

    Вывод:
        Path: Путь к файлу

    Исключения:
        ParameterError: Если формат не поддерживается или у файла нет расширения, а fmt не задан
    """
    target = Path(path)
    if fmt is None and not target.suffix:
        raise ParameterError(f"У файла {target.name} нет расширения: укажите формат в fmt")
    target.write_bytes(dumps(transcript, fmt if fmt is not None else target.suffix))
    return target
