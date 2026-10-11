import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

from ..batch import Log, ignore, log_to_stderr
from ..exceptions import MissingDependencyError, ParameterError
from ..pipeline import Transcriber
from ..result import Transcript
from .manifest import Sample
from .text import has_digits, normalize


def require_jiwer() -> ModuleType:
    """
    Пакет jiwer для подсчета ошибок

    Вывод:
        ModuleType: Модуль jiwer

    Исключения:
        MissingDependencyError: Если jiwer не установлен
    """
    try:
        import jiwer
    except ImportError as error:
        raise MissingDependencyError(
            "Для оценки качества нужен пакет jiwer: pip install 'stt-ru[eval]'"
        ) from error
    module: ModuleType = jiwer
    return module


class Counts(TypedDict):
    words: int
    substitutions: int
    deletions: int
    insertions: int
    chars: int
    char_errors: int


@dataclass(frozen=True)
class FileResult:
    """
    Результат одной записи

    Аргументы:
        audio (str): Путь к аудиофайлу
        reference (str): Эталонный текст
        hypothesis (str): Распознанный текст
        duration (float): Длительность записи, с
        seconds (float): Время распознавания записи, с; у строк разметки одного файла общее
        words (int): Слов в нормализованном эталоне
        substitutions (int): Замен слов
        deletions (int): Пропусков слов
        insertions (int): Вставок слов
        chars (int): Символов в нормализованном эталоне
        char_errors (int): Ошибок символов (замены, пропуски, вставки)
        error (str): Текст ошибки распознавания; None - распознано
        channel (int): Канал из разметки; None - весь текст записи
    """

    audio: str
    reference: str
    hypothesis: str = ""
    duration: float = 0.0
    seconds: float = 0.0
    words: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    chars: int = 0
    char_errors: int = 0
    error: str | None = None
    channel: int | None = None

    @property
    def wer(self) -> float | None:
        """Доля ошибок слов; None - пустой эталон или ошибка распознавания"""
        if self.error is not None or self.words == 0:
            return None
        return (self.substitutions + self.deletions + self.insertions) / self.words

    @property
    def cer(self) -> float | None:
        """Доля ошибок символов; None - пустой эталон или ошибка распознавания"""
        if self.error is not None or self.chars == 0:
            return None
        return self.char_errors / self.chars

    @property
    def digits_mismatch(self) -> bool:
        """Цифры есть только в эталоне или только в непустом распознанном тексте"""
        return (
            self.error is None
            and bool(self.hypothesis.strip())
            and has_digits(self.hypothesis) != has_digits(self.reference)
        )


@dataclass
class ModelReport:
    """
    Оценка модели на разметке

    Атрибуты:
        model (str): Имя модели
        files (list[FileResult]): Результаты записей в порядке разметки
        error (str): Ошибка загрузки модели; None - модель загружена
    """

    model: str
    files: list[FileResult] = field(default_factory=list)
    error: str | None = None

    @property
    def scored(self) -> list[FileResult]:
        """Распознанные записи"""
        return [result for result in self.files if result.error is None]

    @property
    def failed(self) -> list[FileResult]:
        """Записи с ошибкой распознавания"""
        return [result for result in self.files if result.error is not None]

    @property
    def wer(self) -> float | None:
        """Доля ошибок слов по всем распознанным записям; None - нечего считать"""
        words = sum(result.words for result in self.scored)
        errors = sum(r.substitutions + r.deletions + r.insertions for r in self.scored)
        return errors / words if words else None

    @property
    def cer(self) -> float | None:
        """Доля ошибок символов по всем распознанным записям; None - нечего считать"""
        chars = sum(result.chars for result in self.scored)
        return sum(result.char_errors for result in self.scored) / chars if chars else None

    @property
    def speed(self) -> float | None:
        """Во сколько раз распознавание быстрее реального времени; None - нечего считать"""
        files = {result.audio: result for result in self.scored}.values()
        seconds = sum(result.seconds for result in files)
        duration = sum(result.duration for result in files)
        return duration / seconds if seconds > 0 else None

    @property
    def digits_mismatch(self) -> int:
        """Сколько записей, где цифры есть только в эталоне или только в распознанном тексте"""
        return sum(result.digits_mismatch for result in self.files)

    def to_dict(self) -> dict[str, Any]:
        """
        Отчет для сохранения в JSON

        Вывод:
            dict: Сводные метрики и результаты записей с wer и cer
        """
        return {
            "model": self.model,
            "error": self.error,
            "wer": self.wer,
            "cer": self.cer,
            "speed": self.speed,
            "files": len(self.files),
            "failed": len(self.failed),
            "digits_mismatch": self.digits_mismatch,
            "results": [
                {**vars(result), "wer": result.wer, "cer": result.cer} for result in self.files
            ],
        }


def _hypothesis(transcript: Transcript, channel: int | None) -> str:
    segments = transcript.segments
    # Одна дорожка - моно, сведенные или одинаковые каналы: сравнивается весь текст
    if channel is not None and transcript.channels > 1:
        if channel >= transcript.channels:
            raise ParameterError(f"канала {channel} нет: каналов в записи - {transcript.channels}")
        segments = tuple(segment for segment in segments if segment.channel == channel)
    return " ".join(segment.text for segment in segments)


def _counts(jiwer: ModuleType, reference: str, hypothesis: str) -> Counts:
    words = len(reference.split())
    if not reference or not hypothesis:
        char_errors = max(len(reference), len(hypothesis))
        return {
            "words": words,
            "substitutions": 0,
            "deletions": words,
            "insertions": len(hypothesis.split()),
            "chars": len(reference),
            "char_errors": char_errors,
        }
    measure = jiwer.process_words(reference, hypothesis)
    chars = jiwer.process_characters(reference, hypothesis)
    return {
        "words": words,
        "substitutions": measure.substitutions,
        "deletions": measure.deletions,
        "insertions": measure.insertions,
        "chars": len(reference),
        "char_errors": chars.substitutions + chars.deletions + chars.insertions,
    }


def evaluate(
    transcriber: Transcriber,
    samples: Sequence[Sample],
    *,
    log: Log = ignore,
    errors: Log = log_to_stderr,
) -> ModelReport:
    """
    Оценка качества распознавания на разметке

    Описание:
        Эталон и распознанный текст нормализуются (см. normalize), ошибки слов и символов
        считаются расстоянием Левенштейна (jiwer). Для стереозаписи с каналом сравнивается
        текст этого канала; если дорожка одна, канал не учитывается. Файл, который встречается
        в разметке несколько раз (например, по строке на канал), распознается один раз. Ошибка
        записи не останавливает оценку

    Аргументы:
        transcriber (Transcriber): Распознаватель
        samples (list[Sample]): Записи с эталонами
        log (Callable[[str], None]): Ход работы
        errors (Callable[[str], None]): Ошибки записей

    Вывод:
        ModelReport: Метрики модели и результаты записей

    Исключения:
        MissingDependencyError: Если не установлен jiwer
    """
    jiwer = require_jiwer()
    report = ModelReport(transcriber.recognizer.model)
    uses = Counter(sample.audio for sample in samples)
    recognized: dict[Path, tuple[Transcript, float] | str] = {}
    for number, sample in enumerate(samples, 1):
        prefix = f"[{report.model} {number}/{len(samples)}] {sample.audio}"
        if sample.channel is not None:
            prefix += f", канал {sample.channel}"
        outcome = recognized.pop(sample.audio, None) or _recognize(transcriber, sample.audio)
        uses[sample.audio] -= 1
        if uses[sample.audio]:
            recognized[sample.audio] = outcome
        try:
            if isinstance(outcome, str):
                raise ParameterError(outcome)
            transcript, seconds = outcome
            hypothesis = _hypothesis(transcript, sample.channel)
        except ParameterError as error:
            report.files.append(
                FileResult(
                    str(sample.audio), sample.text, error=str(error), channel=sample.channel
                )
            )
            errors(f"{prefix}: ошибка - {error}")
            continue
        result = FileResult(
            str(sample.audio),
            sample.text,
            hypothesis,
            transcript.duration,
            seconds,
            **_counts(jiwer, normalize(sample.text), normalize(hypothesis)),
            channel=sample.channel,
        )
        report.files.append(result)
        log(f"{prefix}: WER {_percent(result.wer)}, CER {_percent(result.cer)}")
    return report


def _recognize(transcriber: Transcriber, path: Path) -> tuple[Transcript, float] | str:
    started = time.perf_counter()
    try:
        transcript = transcriber.transcribe(path)
    except Exception as error:
        # Текст, а не исключение: трассировка держала бы в памяти аудио, пока файл в кэше
        return str(error) or type(error).__name__
    return transcript, time.perf_counter() - started


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f} %"


def format_table(reports: Sequence[ModelReport]) -> str:
    """
    Таблица сравнения моделей

    Аргументы:
        reports (list[ModelReport]): Оценки моделей

    Вывод:
        str: Строки таблицы: модель, WER и CER в процентах, скорость, число записей и ошибок

    Пример использования:
        >>> from stt_ru.evaluation import FileResult, ModelReport, format_table
        >>> result = FileResult("a.wav", "раз два", "раз три", 10.0, 0.5, 2, 1, 0, 0, 7, 3)
        >>> print(format_table([ModelReport("gigaam-v3-e2e-rnnt", [result])]))
        Модель              WER, %  CER, %  Скорость  Записей  Ошибок
        gigaam-v3-e2e-rnnt    50.0    42.9     20.0x        1       0
    """
    header = ("Модель", "WER, %", "CER, %", "Скорость", "Записей", "Ошибок")
    rows = [header]
    for report in reports:
        if report.error is not None:
            rows.append((f"{report.model} (не загружена)", "-", "-", "-", "-", "-"))
            continue
        rows.append(
            (
                report.model,
                "-" if report.wer is None else f"{report.wer * 100:.1f}",
                "-" if report.cer is None else f"{report.cer * 100:.1f}",
                "-" if report.speed is None else f"{report.speed:.1f}x",
                str(len(report.files)),
                str(len(report.failed)),
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = []
    for row in rows:
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)
