import pytest

from stt_ru.exceptions import ParameterError
from stt_ru.vad import VadOptions


def test_defaults():
    assert VadOptions().as_kwargs() == {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 20,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 30,
    }


def test_neg_threshold():
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
        ({"max_speech_duration_s": 0}, "max_speech_duration_s должен быть положительным"),
    ],
)
def test_wrong_values(options, message):
    with pytest.raises(ParameterError, match=message):
        VadOptions(**options)


def test_parameter_error_is_value_error():
    assert issubclass(ParameterError, ValueError)
