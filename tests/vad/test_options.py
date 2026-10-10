import pytest

from stt_ru.exceptions import ParameterError
from stt_ru.vad import VadOptions


def test_defaults():
    assert VadOptions().as_kwargs() == {
        "threshold": 0.5,
        "neg_threshold": 0.35,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 20,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 30,
    }


@pytest.mark.parametrize(
    ("threshold", "neg_threshold"), [(0.45, 0.3), (0.16, 0.01), (0.15, 0.01), (0.1, 0.01)]
)
def test_implicit_neg_threshold(threshold, neg_threshold):
    assert VadOptions(threshold=threshold).as_kwargs()["neg_threshold"] == neg_threshold


def test_explicit_neg_threshold():
    assert VadOptions(neg_threshold=0.2).as_kwargs()["neg_threshold"] == 0.2


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"threshold": 0}, "threshold должен быть от 0 до 1"),
        ({"threshold": 1}, "threshold должен быть от 0 до 1"),
        ({"neg_threshold": 0.6}, "neg_threshold должен быть от 0 до threshold"),
        ({"neg_threshold": 0}, "neg_threshold должен быть от 0 до threshold"),
        ({"min_speech_duration_ms": -1}, "min_speech_duration_ms не может быть отрицательным"),
        ({"min_silence_duration_ms": -1}, "min_silence_duration_ms не может быть отрицательным"),
        ({"speech_pad_ms": -1}, "speech_pad_ms не может быть отрицательным"),
        ({"max_speech_duration_s": 0}, "должен быть больше двух speech_pad_ms"),
        (
            {"max_speech_duration_s": 1, "speech_pad_ms": 500},
            r"max_speech_duration_s \(1 с\) должен быть больше двух speech_pad_ms \(500 мс\)",
        ),
    ],
)
def test_wrong_values(options, message):
    with pytest.raises(ParameterError, match=message):
        VadOptions(**options)


def test_pad_just_below_limit():
    assert VadOptions(max_speech_duration_s=1, speech_pad_ms=499).speech_pad_ms == 499


def test_parameter_error_is_value_error():
    assert issubclass(ParameterError, ValueError)
