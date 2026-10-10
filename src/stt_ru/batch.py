import dataclasses
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ParameterError
from .export import FORMATS, dumps, to_csv, to_jsonl
from .pipeline import Transcriber
from .result import Transcript

AUDIO_SUFFIXES = frozenset(
    {
        ".3gp", ".aac", ".aif", ".aiff", ".amr", ".au", ".avi", ".caf", ".flac", ".m4a",
        ".mkv", ".mov", ".mp3", ".mp4", ".oga", ".ogg", ".opus", ".w64", ".wav", ".webm",
        ".wma",
    }
)  # fmt: skip
SUMMARY_FORMATS = ("csv", "jsonl")

Log = Callable[[str], None]


def log_to_stderr(message: str) -> None:
    """Вывод сообщения в stderr"""
    print(message, file=sys.stderr, flush=True)


def ignore(message: str) -> None:
    """Сообщение не выводится"""


@dataclass(frozen=True)
class Job:
    """
    Файл для распознавания

    Аргументы:
        source (Path): Путь к аудиофайлу
        name (Path): Путь результата без расширения относительно каталога результатов
        label (str): Имя файла в общем файле: путь внутри каталога или имя файла
    """

    source: Path
    name: Path
    label: str


@dataclass
class Report:
    """
    Итог обработки файлов

    Атрибуты:
        done (list[Path]): Распознанные файлы
        skipped (list[Path]): Файлы, у которых результаты уже есть
        failed (dict[Path, str]): Файлы с ошибкой и ее текст
    """

    done: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: dict[Path, str] = field(default_factory=dict)


@dataclass
class WatchReport:
    """
    Итог мониторинга

    Атрибуты:
        done (int): Сколько файлов распознано
        skipped (int): Сколько файлов пропущено: результаты уже были
        failed (dict[Path, str]): Файлы, которые есть в каталогах и последняя попытка
            которых закончилась ошибкой, и текст ошибки
    """

    done: int = 0
    skipped: int = 0
    failed: dict[Path, str] = field(default_factory=dict)


def find_jobs(
    paths: Sequence[str | os.PathLike[str]],
    recursive: bool = False,
    *,
    strict: bool = True,
    log: Log = ignore,
) -> list[Job]:
    """
    Поиск файлов для распознавания

    Описание:
        Файл из аргументов берется с любым расширением, из каталога - только с расширением
        из AUDIO_SUFFIXES, без скрытых файлов и каталогов (имя с точки). Результат файла
        из каталога сохраняет его путь внутри каталога. Один файл, переданный разными
        путями, берется один раз; имена результатов сравниваются без учета регистра

    Аргументы:
        paths (list[str]): Файлы и каталоги
        recursive (bool): Искать файлы и во вложенных каталогах
        strict (bool): Отсутствующий путь и совпадение имен результатов - ошибка;
            False - путь и файл с совпадающим именем пропускаются с сообщением в log
        log (Callable[[str], None]): Сообщения о пропущенных путях и файлах при strict=False

    Вывод:
        list[Job]: Файлы в порядке аргументов, внутри каталога - по имени

    Исключения:
        FileNotFoundError: Если пути нет и strict=True
        ParameterError: Если у двух файлов совпадают имена результатов и strict=True
    """
    jobs: list[Job] = []
    for item in map(Path, paths):
        if item.is_dir():
            for file in sorted(item.rglob("*") if recursive else item.glob("*")):
                relative = file.relative_to(item)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                if file.suffix.lower() in AUDIO_SUFFIXES and file.is_file():
                    jobs.append(Job(file, relative.with_suffix(""), relative.as_posix()))
        elif item.is_file():
            jobs.append(Job(item, Path(item.stem), item.name))
        elif strict:
            raise FileNotFoundError(f"Файл или каталог не найден: {item}")
        else:
            log(f"Файл или каталог не найден: {item}")
    unique: dict[Path, Job] = {}
    for job in jobs:
        unique.setdefault(job.source.resolve(), job)
    taken: dict[str, Path] = {}
    result = []
    for job in unique.values():
        key = job.name.as_posix().casefold()
        if key in taken:
            message = f"У файлов {taken[key]} и {job.source} совпадают имена результатов"
            if strict:
                raise ParameterError(message)
            log(f"{message}: {job.source} пропущен")
            continue
        taken[key] = job.source
        result.append(job)
    return result


def check_options(
    formats: Sequence[str],
    summary: str | os.PathLike[str] | None = None,
    workers: int = 1,
) -> None:
    """
    Проверка параметров пакетной обработки

    Аргументы:
        formats (list[str]): Форматы результатов
        summary (str): Общий файл
        workers (int): Число потоков

    Исключения:
        ParameterError: Если формат не поддерживается, у общего файла не то расширение
            или число потоков не положительно
    """
    unknown = [fmt for fmt in formats if fmt not in FORMATS]
    if unknown:
        raise ParameterError(
            f"Форматы не поддерживаются: {', '.join(unknown)}; допустимые: " + ", ".join(FORMATS)
        )
    if summary is not None and Path(summary).suffix.lower().lstrip(".") not in SUMMARY_FORMATS:
        raise ParameterError(f"Общий файл должен быть CSV или JSONL: {summary}")
    if workers < 1:
        raise ParameterError(f"Число потоков должно быть положительным: {workers}")


def write_atomic(path: Path, content: bytes) -> None:
    """
    Запись файла целиком или никак

    Описание:
        Содержимое пишется во временный скрытый файл рядом и переименовывается: при сбое
        или прерывании файла с неполным содержимым не остается

    Аргументы:
        path (Path): Путь к файлу
        content (bytes): Содержимое
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class Batch:
    """
    Распознавание файлов с записью результатов

    Описание:
        Для каждого файла по порядку: строки в общий файл, результаты в форматах
        (каждый целиком или никак), on_done, удаление исходника. Файл, у которого есть
        все результаты, пропускается; при delete_source его исходник удаляется

    Аргументы:
        transcriber (Transcriber): Распознаватель
        output_dir (str): Каталог результатов; None - без файлов результатов
        formats (list[str]): Форматы результатов из FORMATS
        summary (str): Общий файл CSV или JSONL: создается или дописывается строками файлов,
            распознанных в этом запуске
        overwrite (bool): Распознавать файлы, у которых результаты уже есть
        delete_source (bool): Удалять файл после распознавания и записи результатов
        workers (int): Число файлов, которые распознаются одновременно
        log (Callable[[str], None]): Ход работы
        errors (Callable[[str], None]): Ошибки файлов
        on_done (Callable[[Job, Transcript], None]): Вызывается для каждого распознанного
            файла после записи результатов, в порядке завершения

    Исключения:
        ParameterError: Если параметры не проходят check_options или delete_source задан
            без каталога результатов и общего файла
    """

    def __init__(
        self,
        transcriber: Transcriber,
        output_dir: str | os.PathLike[str] | None,
        formats: Sequence[str],
        *,
        summary: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
        delete_source: bool = False,
        workers: int = 1,
        log: Log = log_to_stderr,
        errors: Log = log_to_stderr,
        on_done: Callable[[Job, Transcript], None] | None = None,
    ) -> None:
        check_options(formats, summary, workers)
        if delete_source and output_dir is None and summary is None:
            raise ParameterError("Удалять исходники можно, только если результаты сохраняются")
        self.transcriber = transcriber
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.formats = list(formats) if output_dir is not None else []
        self.summary = None if summary is None else Path(summary)
        self.overwrite = overwrite
        self.delete_source = delete_source
        self.workers = workers
        self.log = log
        self.errors = errors
        self.on_done = on_done
        self._lock = threading.Lock()

    def outputs(self, job: Job) -> list[Path]:
        """
        Пути результатов файла

        Аргументы:
            job (Job): Файл

        Вывод:
            list[Path]: Путь для каждого формата; пусто без каталога результатов
        """
        if self.output_dir is None:
            return []
        return [
            self.output_dir / job.name.with_name(f"{job.name.name}.{fmt}") for fmt in self.formats
        ]

    def run(self, jobs: Sequence[Job]) -> Report:
        """
        Распознавание файлов

        Описание:
            Ошибка одного файла не останавливает остальные. При прерывании (Ctrl+C) файлы
            из очереди отменяются, а файлы в работе дораспознаются и сохраняются;
            повторное прерывание выходит, не дожидаясь их

        Аргументы:
            jobs (list[Job]): Файлы

        Вывод:
            Report: Распознанные, пропущенные и упавшие файлы
        """
        report = Report()
        pending = []
        for job in jobs:
            outputs = self.outputs(job)
            if outputs and not self.overwrite and all(path.exists() for path in outputs):
                report.skipped.append(job.source)
                self.log(f"{job.source}: пропущен, результаты уже есть")
                if self.delete_source:
                    self._delete(job, report)
            else:
                pending.append(job)
        executor = ThreadPoolExecutor(max_workers=self.workers)
        futures = {executor.submit(self._transcribe, job): job for job in pending}
        counter = iter(range(1, len(pending) + 1))
        try:
            for future in as_completed(futures):
                self._finish(
                    futures.pop(future), future, report, f"[{next(counter)}/{len(pending)}]"
                )
        except KeyboardInterrupt:
            running = [future for future in futures if not future.cancel()]
            if running:
                self.log(
                    f"Прерывание: дожидаюсь файлов в работе ({len(running)}), "
                    "Ctrl+C еще раз - выйти сразу"
                )
                wait(running)
                for future in running:
                    self._finish(futures.pop(future), future, report, "[прерывание]")
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return report

    def _transcribe(self, job: Job) -> tuple[Transcript, float]:
        started = time.perf_counter()
        transcript = self.transcriber.transcribe(job.source)
        return transcript, time.perf_counter() - started

    def _finish(
        self, job: Job, future: "Future[tuple[Transcript, float]]", report: Report, prefix: str
    ) -> None:
        try:
            transcript, seconds = future.result()
            self._save(job, transcript)
        except Exception as error:
            report.failed[job.source] = str(error)
            self.errors(f"{prefix} {job.source}: ошибка - {error}")
            return
        if self.on_done is not None:
            self.on_done(job, transcript)
        report.done.append(job.source)
        self.log(
            f"{prefix} {job.source}: фрагментов {len(transcript.segments)}, "
            f"{transcript.duration:.1f} с записи за {seconds:.1f} с"
        )
        if self.delete_source:
            self._delete(job, report)

    def _save(self, job: Job, transcript: Transcript) -> None:
        if self.summary is not None:
            self._append_summary(dataclasses.replace(transcript, source=job.label))
        for path, fmt in zip(self.outputs(job), self.formats, strict=True):
            write_atomic(path, dumps(transcript, fmt))

    def _delete(self, job: Job, report: Report) -> None:
        try:
            job.source.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            report.failed[job.source] = f"не удалось удалить исходник: {error}"
            self.errors(f"{job.source}: не удалось удалить исходник - {error}")

    def _append_summary(self, transcript: Transcript) -> None:
        assert self.summary is not None
        with self._lock:
            self.summary.parent.mkdir(parents=True, exist_ok=True)
            if self.summary.suffix.lower() == ".jsonl":
                content = to_jsonl(transcript).encode("utf-8")
            elif self.summary.exists() and self.summary.stat().st_size > 0:
                content = to_csv(transcript, header=False).encode("utf-8")
            else:
                content = dumps(transcript, "csv")
            with self.summary.open("ab") as file:
                file.write(content)


def watch(
    batch: Batch,
    directories: Sequence[str | os.PathLike[str]],
    *,
    recursive: bool = False,
    interval: float = 10.0,
    min_age: float = 10.0,
    rounds: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> WatchReport:
    """
    Мониторинг каталогов: распознавание новых файлов

    Описание:
        Каталоги просматриваются каждые interval секунд; файл берется в работу, когда
        не менялся min_age секунд. Файл с ошибкой не повторяется, пока не изменится.
        Пропавший каталог и файлы с совпадающими именами результатов не останавливают
        мониторинг: о них пишется сообщение в batch.errors

    Аргументы:
        batch (Batch): Распознавание с записью результатов
        directories (list[str]): Каталоги
        recursive (bool): Просматривать и вложенные каталоги
        interval (float): Пауза между просмотрами, с, больше 0
        min_age (float): Сколько секунд файл не должен меняться, не меньше 0
        rounds (int): Число просмотров; None - до прерывания
        sleep (Callable[[float], None]): Функция паузы
        clock (Callable[[], float]): Текущее время, с

    Вывод:
        WatchReport: Итог всех просмотров

    Исключения:
        ParameterError: Если interval не положителен или min_age отрицателен
    """
    check_watch(interval, min_age)
    total = WatchReport()
    attempts: dict[Path, tuple[float, int]] = {}
    reported: set[str] = set()

    def note(message: str) -> None:
        if message not in reported:
            reported.add(message)
            batch.errors(message)

    count = 0
    while rounds is None or count < rounds:
        count += 1
        now = clock()
        try:
            jobs = find_jobs(directories, recursive, strict=False, log=note)
        except OSError as error:
            note(f"Не удалось просмотреть каталоги: {error}")
            jobs = []
        present = {job.source for job in jobs}
        attempts = {path: signature for path, signature in attempts.items() if path in present}
        total.failed = {path: text for path, text in total.failed.items() if path in present}
        ready = []
        for job in jobs:
            try:
                stat = job.source.stat()
            except FileNotFoundError:
                continue
            signature = (stat.st_mtime, stat.st_size)
            if now - stat.st_mtime >= min_age and attempts.get(job.source) != signature:
                ready.append((job, signature))
        report = batch.run([job for job, _ in ready])
        for job, signature in ready:
            attempts[job.source] = signature
        total.done += len(report.done)
        total.skipped += len(report.skipped)
        for path in report.done:
            total.failed.pop(path, None)
        total.failed.update(report.failed)
        if rounds is None or count < rounds:
            sleep(interval)
    return total


def check_watch(interval: float, min_age: float) -> None:
    """
    Проверка параметров мониторинга

    Аргументы:
        interval (float): Пауза между просмотрами, с
        min_age (float): Сколько секунд файл не должен меняться

    Исключения:
        ParameterError: Если interval не положителен или min_age отрицателен
    """
    if interval <= 0:
        raise ParameterError(f"Интервал просмотра должен быть положительным: {interval}")
    if min_age < 0:
        raise ParameterError(f"Время без изменений не может быть отрицательным: {min_age}")


def summary_line(report: Report | WatchReport) -> str:
    """
    Итоговая строка отчета

    Аргументы:
        report (Report): Отчет

    Вывод:
        str: Сколько файлов распознано, пропущено и с ошибкой
    """
    done = report.done if isinstance(report.done, int) else len(report.done)
    skipped = report.skipped if isinstance(report.skipped, int) else len(report.skipped)
    failed = len(report.failed)
    return (
        f"Готово: распознано {done} из {done + skipped + failed}, пропущено {skipped}, "
        f"с ошибкой {failed}"
    )
