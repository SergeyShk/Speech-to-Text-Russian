import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..batch import AUDIO_SUFFIXES
from ..exceptions import ParameterError


@dataclass(frozen=True)
class Sample:
    """
    Запись с эталонным текстом

    Аргументы:
        audio (Path): Путь к аудиофайлу
        text (str): Эталонный текст
        channel (int): Канал стереозаписи, с которым сравнивается эталон; None - все каналы
    """

    audio: Path
    text: str
    channel: int | None = None


def load_manifest(path: str | os.PathLike[str]) -> list[Sample]:
    """
    Чтение разметки для оценки качества

    Описание:
        Разметка - CSV с колонками audio, text и необязательной channel, JSON Lines с полями
        audio, text и channel или каталог, где у каждого аудиофайла есть текстовый файл
        с эталоном и тем же именем (call.wav и call.txt). Пути к аудио - относительно
        файла разметки. Канал задается для стереозаписей, где канал - говорящий

    Аргументы:
        path (str): Файл разметки или каталог

    Вывод:
        list[Sample]: Записи в порядке разметки

    Исключения:
        FileNotFoundError: Если нет файла разметки или аудиофайла из нее
        ParameterError: Если разметка не в UTF-8 или пуста, в ней нет нужных полей,
            эталон пуст или канал - не целое неотрицательное число
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Разметка не найдена: {source}")
    if not source.is_dir() and source.suffix.lower() not in (".csv", ".jsonl"):
        raise ParameterError(f"Разметка должна быть CSV, JSONL или каталогом: {source}")
    if source.is_dir():
        samples = _from_directory(source)
    elif source.suffix.lower() == ".jsonl":
        samples = _from_jsonl(source)
    else:
        samples = _from_csv(source)
    if not samples:
        raise ParameterError(f"В разметке нет записей: {source}")
    for sample in samples:
        if not sample.audio.is_file():
            raise FileNotFoundError(f"Аудиофайл из разметки не найден: {sample.audio}")
    return samples


def _sample(base: Path, row: dict[str, object], where: str) -> Sample:
    audio, text, channel = row.get("audio"), row.get("text"), row.get("channel")
    if not isinstance(audio, str) or not audio.strip():
        raise ParameterError(f"{where}: нет пути к аудио в поле audio")
    if not isinstance(text, str) or not text.strip():
        raise ParameterError(f"{where}: пустой эталон в поле text")
    if channel in (None, ""):
        number = None
    elif isinstance(channel, int) and not isinstance(channel, bool):
        number = channel
    elif isinstance(channel, str) and channel.strip().isdigit():
        number = int(channel)
    else:
        number = -1
    if number is not None and number < 0:
        raise ParameterError(f"{where}: канал должен быть целым неотрицательным числом: {channel}")
    return Sample(base / audio.strip(), text, number)


def _not_utf8(path: Path) -> ParameterError:
    return ParameterError(f"{path}: файл не в кодировке UTF-8, пересохраните его в UTF-8")


def _from_csv(path: Path) -> list[Sample]:
    # Эталон многочасовой записи длиннее стандартного предела поля csv в 128 КБ
    limit = csv.field_size_limit(2**31 - 1)
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = {"audio", "text"} - set(reader.fieldnames or [])
            if missing:
                raise ParameterError(f"{path}: нет колонок {', '.join(sorted(missing))}")
            return [
                _sample(path.parent, dict(row), f"{path}, строка {reader.line_num}")
                for row in reader
            ]
    except UnicodeDecodeError as error:
        raise _not_utf8(path) from error
    finally:
        csv.field_size_limit(limit)


def _from_jsonl(path: Path) -> list[Sample]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise _not_utf8(path) from error
    samples = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"{path}, строка {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ParameterError(f"{where}: неверный JSON - {error}") from error
        if not isinstance(row, dict):
            raise ParameterError(f"{where}: строка должна быть объектом JSON")
        samples.append(_sample(path.parent, row, where))
    return samples


def _from_directory(path: Path) -> list[Sample]:
    samples = []
    missing = []
    for audio in sorted(path.iterdir()):
        if audio.name.startswith(".") or audio.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        reference = audio.with_suffix(".txt")
        if not reference.is_file():
            missing.append(audio.name)
            continue
        try:
            text = reference.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            raise _not_utf8(reference) from error
        if not text.strip():
            raise ParameterError(f"{reference}: пустой эталон")
        samples.append(Sample(audio, text))
    if missing:
        raise ParameterError(
            f"Нет эталонов (файлов .txt с тем же именем) для {len(missing)} файлов: "
            + ", ".join(missing[:5])
            + (" и других" if len(missing) > 5 else "")
        )
    return samples
