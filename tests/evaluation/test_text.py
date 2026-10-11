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
        ("Вои\u0306на и\u0306од", "война йод"),
        ("Е\u0308ж", "еж"),
        ("ка\u0301ша, мо\u0301локо\u0301", "каша молоко"),
        ("пере\u00adнос и про\u200bбел\ufeff", "перенос и пробел"),
        ("Café Cafe\u0301", "café café"),
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
