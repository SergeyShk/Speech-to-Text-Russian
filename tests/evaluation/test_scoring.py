import sys
from types import SimpleNamespace

import pytest

from stt_ru import Segment, Transcript
from stt_ru.evaluation import (
    FileResult,
    ModelReport,
    Sample,
    evaluate,
    format_table,
    require_jiwer,
)
from stt_ru.exceptions import AudioError, MissingDependencyError


class FakeTranscriber:
    """Распознает по таблице: имя файла - фрагменты; нет в таблице - ошибка чтения"""

    def __init__(self, texts, channels=2):
        self.recognizer = SimpleNamespace(model="fake")
        self.texts = texts
        self.channels = channels

    def transcribe(self, path):
        if path.name not in self.texts:
            raise AudioError(f"Не удалось прочитать {path.name}")
        segments = tuple(
            Segment(0.0, 1.0, text, channel) for channel, text in self.texts[path.name]
        )
        return Transcript(segments, 4.0, self.channels, "fake", path.name)


def test_require_jiwer():
    assert require_jiwer().__name__ == "jiwer"


def test_missing_jiwer(monkeypatch):
    monkeypatch.setitem(sys.modules, "jiwer", None)
    with pytest.raises(MissingDependencyError, match=r"pip install 'stt-ru\[eval\]'"):
        require_jiwer()
    with pytest.raises(ImportError):
        evaluate(FakeTranscriber({}), [])


class TestEvaluate:
    def test_counts(self, tmp_path):
        transcriber = FakeTranscriber(
            {"a.wav": [(0, "Привет, как дела?"), (1, "Хорошо.")], "b.wav": [(None, "Раз два")]},
            channels=1,
        )
        samples = [
            Sample(tmp_path / "a.wav", "привет как у тебя дела хорошо"),
            Sample(tmp_path / "b.wav", "Раз, два, три."),
        ]
        report = evaluate(transcriber, samples)
        first, second = report.files
        assert report.model == "fake"
        assert first.hypothesis == "Привет, как дела? Хорошо."
        assert (first.words, first.substitutions, first.deletions, first.insertions) == (
            6,
            0,
            2,
            0,
        )
        assert first.chars == len("привет как у тебя дела хорошо")
        assert first.char_errors == len("у тебя ")
        assert first.wer == pytest.approx(2 / 6)
        assert (second.words, second.deletions, second.duration) == (3, 1, 4.0)
        assert second.cer == pytest.approx(4 / 11)
        assert report.wer == pytest.approx(3 / 9)
        assert report.cer == pytest.approx((7 + 4) / (29 + 11))
        assert report.speed is not None and report.speed > 0

    def test_channel(self, tmp_path):
        transcriber = FakeTranscriber({"a.wav": [(0, "Алло."), (1, "Добрый день."), (0, "Да.")]})
        samples = [
            Sample(tmp_path / "a.wav", "алло да", 0),
            Sample(tmp_path / "a.wav", "добрый день", 1),
            Sample(tmp_path / "a.wav", "алло добрый день да"),
        ]
        report = evaluate(transcriber, samples)
        assert [result.hypothesis for result in report.files] == [
            "Алло. Да.",
            "Добрый день.",
            "Алло. Добрый день. Да.",
        ]
        assert report.wer == 0

    def test_channel_of_single_track(self, tmp_path):
        transcriber = FakeTranscriber({"a.wav": [(None, "Алло.")]}, channels=1)
        report = evaluate(transcriber, [Sample(tmp_path / "a.wav", "алло", 3)])
        assert report.files[0].hypothesis == "Алло."

    def test_missing_channel(self, tmp_path):
        transcriber = FakeTranscriber({"a.wav": [(0, "Алло."), (1, "Да.")]})
        messages = []
        report = evaluate(
            transcriber, [Sample(tmp_path / "a.wav", "да", 2)], errors=messages.append
        )
        assert report.files[0].error == "канала 2 нет: каналов в записи - 2"
        assert messages == [
            f"[fake 1/1] {tmp_path / 'a.wav'}, канал 2: ошибка - канала 2 нет: каналов в записи - 2"
        ]

    def test_failed_file(self, tmp_path):
        transcriber = FakeTranscriber({"b.wav": [(0, "Да.")]})
        progress, errors = [], []
        samples = [Sample(tmp_path / "a.wav", "нет"), Sample(tmp_path / "b.wav", "да")]
        report = evaluate(transcriber, samples, log=progress.append, errors=errors.append)
        assert [result.error for result in report.files] == ["Не удалось прочитать a.wav", None]
        assert report.failed == [report.files[0]] and report.scored == [report.files[1]]
        assert report.wer == 0
        assert errors == [f"[fake 1/2] {tmp_path / 'a.wav'}: ошибка - Не удалось прочитать a.wav"]
        assert progress == [f"[fake 2/2] {tmp_path / 'b.wav'}: WER 0.0 %, CER 0.0 %"]

    def test_failed_file_to_stderr_by_default(self, tmp_path, capsys):
        evaluate(FakeTranscriber({}), [Sample(tmp_path / "a.wav", "нет")])
        assert "ошибка - Не удалось прочитать a.wav" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("reference", "hypothesis", "counts"),
        [
            ("да нет", "", (2, 0, 2, 0, 6, 6)),
            ("...", "Да, нет.", (0, 0, 0, 2, 0, 6)),
            ("—", "", (0, 0, 0, 0, 0, 0)),
        ],
    )
    def test_empty_text(self, tmp_path, reference, hypothesis, counts):
        texts = {"a.wav": [(None, hypothesis)] if hypothesis else []}
        report = evaluate(
            FakeTranscriber(texts, channels=1), [Sample(tmp_path / "a.wav", reference)]
        )
        result = report.files[0]
        fields = ("words", "substitutions", "deletions", "insertions", "chars", "char_errors")
        assert tuple(getattr(result, name) for name in fields) == counts
        assert report.wer == (1.0 if counts[0] else None)


class TestFileResult:
    def test_metrics(self):
        result = FileResult("a.wav", "раз два", "раз три", 10.0, 0.5, 2, 1, 0, 0, 7, 3)
        assert result.wer == 0.5
        assert result.cer == pytest.approx(3 / 7)

    def test_error(self):
        result = FileResult("a.wav", "15 минут", error="нет файла")
        assert result.wer is None and result.cer is None
        assert not result.digits_mismatch

    @pytest.mark.parametrize(
        ("reference", "hypothesis", "expected"),
        [
            ("пятнадцать минут", "15 минут", True),
            ("15 минут", "15 минут", False),
            ("да", "да", False),
        ],
    )
    def test_digits_mismatch(self, reference, hypothesis, expected):
        assert FileResult("a.wav", reference, hypothesis).digits_mismatch is expected


class TestModelReport:
    def test_empty(self):
        report = ModelReport("fake")
        assert (report.wer, report.cer, report.speed, report.digits_mismatch) == (
            None,
            None,
            None,
            0,
        )

    def test_to_dict(self):
        files = [
            FileResult("a.wav", "пять минут", "5 минут", 6.0, 2.0, 2, 1, 0, 0, 10, 4),
            FileResult("b.wav", "да", error="нет файла"),
        ]
        document = ModelReport("fake", files).to_dict()
        assert document == {
            "model": "fake",
            "error": None,
            "wer": 0.5,
            "cer": 0.4,
            "speed": 3.0,
            "files": 2,
            "failed": 1,
            "digits_mismatch": 1,
            "results": [
                {
                    "audio": "a.wav",
                    "reference": "пять минут",
                    "hypothesis": "5 минут",
                    "duration": 6.0,
                    "seconds": 2.0,
                    "words": 2,
                    "substitutions": 1,
                    "deletions": 0,
                    "insertions": 0,
                    "chars": 10,
                    "char_errors": 4,
                    "error": None,
                    "wer": 0.5,
                    "cer": 0.4,
                },
                {
                    "audio": "b.wav",
                    "reference": "да",
                    "hypothesis": "",
                    "duration": 0.0,
                    "seconds": 0.0,
                    "words": 0,
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": 0,
                    "chars": 0,
                    "char_errors": 0,
                    "error": "нет файла",
                    "wer": None,
                    "cer": None,
                },
            ],
        }


def test_format_table():
    scored = ModelReport(
        "t-tech/t-one", [FileResult("a.wav", "да", "да", 3.0, 0.0, 1, 0, 0, 0, 2, 0)]
    )
    failed = ModelReport("gigaam-v3-ctc", [FileResult("a.wav", "да", error="нет файла")])
    broken = ModelReport("nemo-x", error="нет модели")
    assert format_table([scored, failed, broken]).splitlines() == [
        "Модель                 WER, %  CER, %  Скорость  Записей  Ошибок",
        "t-tech/t-one              0.0     0.0         -        1       0",
        "gigaam-v3-ctc               -       -         -        1       1",
        "nemo-x (не загружена)       -       -         -        -       -",
    ]
