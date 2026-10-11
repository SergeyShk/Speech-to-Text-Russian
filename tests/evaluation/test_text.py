import pytest

from stt_ru.evaluation import has_digits, normalize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Привет, Мир!", "привет мир"),
        ("Ещё ЁЖ", "еще еж"),
        ("кто-то\tсказал\n«да»", "кто то сказал да"),
        ("snake_case и 3,5 %", "snake case и 3 5"),
        ("  ...  ", ""),
        ("", ""),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("в 15:00", True), ("пятнадцать", False), ("", False), ("٣", True)]
)
def test_has_digits(text, expected):
    assert has_digits(text) is expected
