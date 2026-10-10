import csv
import io
import json
import subprocess

import imageio_ffmpeg
import pysubs2
import pytest

from stt_ru import Segment, Transcript
from stt_ru.exceptions import ParameterError
from stt_ru.export import (
    FORMATS,
    dumps,
    render,
    save,
    to_ass,
    to_csv,
    to_json,
    to_jsonl,
    to_srt,
    to_txt,
    to_vtt,
)

LEGACY = (
    Segment(0.78, 7.0, "наталья благодарю вас за ожиданьем", 1),
    Segment(7.0, 8.86, "постарайтесь пожалуйста омовением", 1),
    Segment(7.46, 8.32, "пять", 0),
    Segment(9.4, 12.62, "я в свою очередь с вами прощайте", 1),
    Segment(12.77, 13.77, "бом", 0),
)
LEGACY_DIALOGUES = [
    "Dialogue: 0,0:00:00.78,0:00:07.00,Default,Канал 1,0,0,0,,наталья благодарю вас за ожиданьем",
    "Dialogue: 0,0:00:07.00,0:00:08.86,Default,Канал 1,0,0,0,,постарайтесь пожалуйста омовением",
    "Dialogue: 0,0:00:07.46,0:00:08.32,Default,Канал 0,0,0,0,,пять",
    "Dialogue: 0,0:00:09.40,0:00:12.62,Default,Канал 1,0,0,0,,я в свою очередь с вами прощайте",
    "Dialogue: 0,0:00:12.77,0:00:13.77,Default,Канал 0,0,0,0,,бом",
]
LEGACY_CSV = [
    "example.wav,780,7000,Канал 1,наталья благодарю вас за ожиданьем",
    "example.wav,7000,8860,Канал 1,постарайтесь пожалуйста омовением",
    "example.wav,7460,8320,Канал 0,пять",
    "example.wav,9400,12620,Канал 1,я в свою очередь с вами прощайте",
    "example.wav,12770,13770,Канал 0,бом",
]


@pytest.fixture
def stereo():
    segments = (
        Segment(0.994, 9.054, "Наталья, благодарю вас за ожидание.", 1),
        Segment(3.49, 3.934, "Угу", 0),
    )
    return Transcript(segments, 15.75, 2, "gigaam-v3-e2e-rnnt", "call.wav")


@pytest.fixture
def mixed():
    return Transcript((Segment(0.5, 1.5, "Алло."),), 2.0, 1, "m", None)


@pytest.fixture
def empty():
    return Transcript((), 1.0, 1, "m", "silence.wav")


@pytest.fixture
def legacy():
    return Transcript(LEGACY, 15.75, 2, "m", "example.wav")


def test_formats():
    assert FORMATS == ("txt", "json", "jsonl", "srt", "vtt", "ass", "csv")


class TestTxt:
    def test_speakers(self, stereo):
        assert to_txt(stereo) == "Канал 1: Наталья, благодарю вас за ожидание.\nКанал 0: Угу\n"

    def test_mixed(self, mixed):
        assert to_txt(mixed) == "Алло.\n"

    def test_empty(self, empty):
        assert to_txt(empty) == ""


class TestJson:
    def test_document(self, stereo):
        text = to_json(stereo)
        assert "Наталья" in text
        assert json.loads(text) == {
            "source": "call.wav",
            "model": "gigaam-v3-e2e-rnnt",
            "duration": 15.75,
            "channels": 2,
            "segments": [
                {
                    "start": 0.994,
                    "end": 9.054,
                    "channel": 1,
                    "speaker": "Канал 1",
                    "text": "Наталья, благодарю вас за ожидание.",
                },
                {"start": 3.49, "end": 3.934, "channel": 0, "speaker": "Канал 0", "text": "Угу"},
            ],
        }

    def test_mixed(self, mixed):
        document = json.loads(to_json(mixed))
        assert document["source"] is None
        assert document["segments"][0]["channel"] is None
        assert document["segments"][0]["speaker"] is None

    def test_empty(self, empty):
        assert json.loads(to_json(empty))["segments"] == []


class TestJsonl:
    def test_lines(self, stereo):
        lines = to_jsonl(stereo).splitlines()
        assert [json.loads(line) for line in lines] == [
            {
                "source": "call.wav",
                "start": 0.994,
                "end": 9.054,
                "channel": 1,
                "speaker": "Канал 1",
                "text": "Наталья, благодарю вас за ожидание.",
            },
            {
                "source": "call.wav",
                "start": 3.49,
                "end": 3.934,
                "channel": 0,
                "speaker": "Канал 0",
                "text": "Угу",
            },
        ]
        assert "Наталья" in lines[0]

    def test_empty(self, empty):
        assert to_jsonl(empty) == ""


class TestSrt:
    def test_text(self, stereo):
        assert to_srt(stereo) == (
            "1\n00:00:00,994 --> 00:00:09,054\nКанал 1: Наталья, благодарю вас за ожидание.\n\n"
            "2\n00:00:03,490 --> 00:00:03,934\nКанал 0: Угу\n\n"
        )

    def test_parses(self, stereo):
        events = pysubs2.SSAFile.from_string(to_srt(stereo), format_="srt").events
        assert [(e.start, e.end, e.plaintext) for e in events] == [
            (994, 9054, "Канал 1: Наталья, благодарю вас за ожидание."),
            (3490, 3934, "Канал 0: Угу"),
        ]

    def test_mixed(self, mixed):
        assert to_srt(mixed) == "1\n00:00:00,500 --> 00:00:01,500\nАлло.\n\n"

    @pytest.mark.parametrize(
        ("seconds", "clock"),
        [
            (0.0, "00:00:00,000"),
            (59.9996, "00:01:00,000"),
            (3599.9994, "00:59:59,999"),
            (36000.5, "10:00:00,500"),
            (-0.001, "00:00:00,000"),
        ],
    )
    def test_clock(self, seconds, clock):
        transcript = Transcript((Segment(seconds, seconds, "x"),), 1.0, 1, "m")
        assert to_srt(transcript).splitlines()[1] == f"{clock} --> {clock}"

    def test_empty(self, empty):
        assert to_srt(empty) == ""


class TestVtt:
    def test_text(self, stereo):
        assert to_vtt(stereo) == (
            "WEBVTT\n\n"
            "00:00:00.994 --> 00:00:09.054\n"
            "<v Канал 1>Канал 1: Наталья, благодарю вас за ожидание.\n\n"
            "00:00:03.490 --> 00:00:03.934\n<v Канал 0>Канал 0: Угу\n\n"
        )

    def test_parses(self, stereo):
        events = pysubs2.SSAFile.from_string(to_vtt(stereo), format_="vtt").events
        assert [(e.start, e.end, e.plaintext) for e in events] == [
            (994, 9054, "Канал 1: Наталья, благодарю вас за ожидание."),
            (3490, 3934, "Канал 0: Угу"),
        ]

    def test_ffmpeg_shows_speaker(self, stereo, tmp_path):
        path = save(stereo, tmp_path / "call.vtt")
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "srt",
            "-",
        ]
        output = subprocess.run(command, capture_output=True, check=True).stdout.decode("utf-8")
        assert "Канал 1: Наталья, благодарю вас за ожидание." in output
        assert "Канал 0: Угу" in output

    def test_mixed(self, mixed):
        assert to_vtt(mixed).splitlines()[3] == "Алло."

    def test_escape(self):
        transcript = Transcript((Segment(0.0, 1.0, "a < b & c > d"),), 1.0, 1, "m")
        assert to_vtt(transcript).splitlines()[3] == "a &lt; b &amp; c &gt; d"

    def test_empty(self, empty):
        assert to_vtt(empty) == "WEBVTT\n\n"


class TestAss:
    def test_legacy_events(self, legacy):
        lines = to_ass(legacy).splitlines()
        assert lines[-5:] == LEGACY_DIALOGUES
        assert "Audio File: example.wav" in lines
        assert "Title: Default Aegisub file" in lines

    def test_parses(self, legacy):
        subtitles = pysubs2.SSAFile.from_string(to_ass(legacy), format_="ass")
        assert subtitles.aegisub_project["Audio File"] == "example.wav"
        assert [(e.start, e.end, e.name, e.text) for e in subtitles.events] == [
            (round(s.start * 1000), round(s.end * 1000), s.speaker, s.text) for s in LEGACY
        ]
        assert subtitles.styles["Default"].fontname == "Arial"

    def test_centiseconds(self):
        transcript = Transcript((Segment(0.994, 3599.996, "x"),), 1.0, 1, "m")
        assert "Dialogue: 0,0:00:00.99,1:00:00.00,Default,,0,0,0,,x" in to_ass(transcript)

    def test_newline(self):
        transcript = Transcript((Segment(0.0, 1.0, "раз\nдва"),), 1.0, 1, "m")
        assert to_ass(transcript).splitlines()[-1].endswith(",,раз\\Nдва")

    def test_no_source(self, mixed):
        assert "Audio File: " in to_ass(mixed).splitlines()

    def test_empty(self, empty):
        assert to_ass(empty).endswith(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )


class TestCsv:
    def test_legacy_rows(self, legacy):
        assert to_csv(legacy).splitlines() == ["Audio File,Start,End,Name,Text", *LEGACY_CSV]

    def test_without_header(self, legacy):
        assert to_csv(legacy, header=False).splitlines() == LEGACY_CSV

    def test_quoting(self):
        transcript = Transcript((Segment(0.0, 1.0, 'Да, "конечно".', 0),), 1.0, 2, "m", "a,b.wav")
        (row,) = csv.reader(io.StringIO(to_csv(transcript, header=False)))
        assert row == ["a,b.wav", "0", "1000", "Канал 0", 'Да, "конечно".']

    def test_mixed(self, mixed):
        assert to_csv(mixed, header=False) == ",500,1500,,Алло.\n"

    def test_empty(self, empty):
        assert to_csv(empty) == "Audio File,Start,End,Name,Text\n"


class TestRenderAndSave:
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_render(self, stereo, fmt):
        assert render(stereo, fmt) == render(stereo, f".{fmt.upper()}")

    def test_unknown_format(self, stereo):
        with pytest.raises(ParameterError, match="Формат docx не поддерживается, допустимые: txt"):
            render(stereo, "docx")

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_save_by_suffix(self, stereo, tmp_path, fmt):
        path = save(stereo, tmp_path / f"call.{fmt}")
        assert path == tmp_path / f"call.{fmt}"
        data = path.read_bytes()
        assert b"\r" not in data
        assert data == dumps(stereo, fmt)
        assert data.startswith(b"\xef\xbb\xbf") == (fmt == "csv")

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_dumps(self, stereo, fmt):
        content = dumps(stereo, fmt)
        expected = render(stereo, fmt).encode("utf-8")
        assert content == (b"\xef\xbb\xbf" + expected if fmt == "csv" else expected)

    def test_dumps_unknown_format(self, stereo):
        with pytest.raises(ParameterError, match="Формат pdf не поддерживается"):
            dumps(stereo, "pdf")

    def test_save_explicit_format(self, stereo, tmp_path):
        path = save(stereo, tmp_path / "call.out", "srt")
        assert path.read_text(encoding="utf-8") == to_srt(stereo)

    def test_save_uppercase_suffix(self, stereo, tmp_path):
        assert save(stereo, tmp_path / "CALL.SRT").read_text(encoding="utf-8") == to_srt(stereo)

    def test_save_without_suffix(self, stereo, tmp_path):
        with pytest.raises(ParameterError, match="У файла call нет расширения: укажите формат"):
            save(stereo, tmp_path / "call")

    def test_save_unknown_suffix(self, stereo, tmp_path):
        with pytest.raises(ParameterError, match=r"Формат \.docx не поддерживается"):
            save(stereo, tmp_path / "call.docx")

    def test_transcript_save(self, stereo, tmp_path):
        path = stereo.save(tmp_path / "call.vtt")
        assert path.read_text(encoding="utf-8") == to_vtt(stereo)
        assert stereo.save(tmp_path / "call.out", fmt="txt").read_text(encoding="utf-8") == (
            to_txt(stereo)
        )
