import re
import unicodedata

_SEPARATORS = re.compile(r"[\W_]+")
_DIGITS = re.compile(r"\d")
# Несамостоятельные знаки (Mn) и невидимые символы форматирования (Cf)
_DROPPED = frozenset(("Mn", "Cf"))


def normalize(text: str) -> str:
    """
    Нормализация текста перед подсчетом ошибок

    Описание:
        Нижний регистр, «ё» как «е», все, кроме букв и цифр, - пробел, пробелы схлопываются.
        Буквы, записанные с отдельными диакритическими знаками, собираются (NFC); оставшиеся
        знаки (ударения) и невидимые символы (мягкий перенос, пробел нулевой ширины)
        удаляются. Числа не переводятся: цифры и слова считаются разными словами

    Аргументы:
        text (str): Текст

    Вывод:
        str: Слова через пробел

    Пример использования:
        >>> from stt_ru.evaluation import normalize
        >>> normalize("Ёлка, кто-то   сказал: «15 мину\u0301т»!")
        'елка кто то сказал 15 минут'
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(char for char in text if unicodedata.category(char) not in _DROPPED)
    return " ".join(_SEPARATORS.sub(" ", text.lower().replace("ё", "е")).split())


def has_digits(text: str) -> bool:
    """
    Есть ли в тексте цифры

    Аргументы:
        text (str): Текст

    Вывод:
        bool: В тексте есть хотя бы одна цифра

    Пример использования:
        >>> from stt_ru.evaluation import has_digits
        >>> has_digits("в 15 часов"), has_digits("в пятнадцать часов")
        (True, False)
    """
    return _DIGITS.search(text) is not None
