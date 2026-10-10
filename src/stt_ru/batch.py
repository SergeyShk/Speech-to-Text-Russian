import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
OnDone = Callable[["Job", Transcript], None]


def log_to_stderr(message: str) -> None:
    """Вывод сообщения в stderr"""
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Job:
    """
    Файл для распознавания

    Аргументы:
        source (Path): Путь к аудиофайлу
        name (Path): Путь результата без расширения относительно каталога результатов
    """

    source: Path
    name: Path


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


def find_jobs(paths: Sequence[str | os.PathLike[str]], recursive: bool = False) -> list[Job]:
    """
    Поиск файлов для распознавания

    Описание:
        Файл из аргументов берется с любым расширением, из каталога - только с расширением
        из AUDIO_SUFFIXES. Результат файла из каталога сохраняет его путь внутри каталога

    Аргументы:
        paths (list[str]): Файлы и каталоги
        recursive (bool): Искать файлы и во вложенных каталогах

    Вывод:
        list[Job]: Файлы в порядке аргументов, внутри каталога - по имени

    Исключения:
        FileNotFoundError: Если пути нет
        ParameterError: Если у двух файлов совпадают пути результатов
    """
    jobs: list[Job] = []
    for item in map(Path, paths):
        if item.is_dir():
            files = item.rglob("*") if recursive else item.glob("*")
            jobs += [
                Job(file, file.relative_to(item).with_suffix(""))
                for file in sorted(files)
                if file.is_file() and file.suffix.lower() in AUDIO_SUFFIXES
            ]
        elif item.is_file():
            jobs.append(Job(item, Path(item.stem)))
        else:
            raise FileNotFoundError(f"Файл или каталог не найден: {item}")
    names: dict[Path, Path] = {}
    for job in jobs:
        if job.name in names and names[job.name] != job.source:
            raise ParameterError(
                f"У файлов {names[job.name]} и {job.source} совпадают имена результатов"
            )
        names[job.name] = job.source
    unique = {job.source.resolve(): job for job in jobs}
    return list(unique.values())


class Batch:
    """
    Распознавание файлов с записью результатов

    Аргументы:
        transcriber (Transcriber): Распознаватель
        output_dir (str): Каталог результатов
        formats (list[str]): Форматы результатов из FORMATS
        summary (str): Общий файл CSV или JSONL для всех файлов: создается или дописывается
        overwrite (bool): Распознавать файлы, у которых результаты уже есть
        delete_source (bool): Удалять файл после успешного распознавания
        workers (int): Число файлов, которые распознаются одновременно
        log (Callable[[str], None]): Вывод сообщений о ходе работы
        on_done (Callable[[Job, Transcript], None]): Вызывается для каждого распознанного
            файла в порядке завершения

    Исключения:
        ParameterError: Если формат не поддерживается, у общего файла не то расширение
            или число потоков не положительно
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
        on_done: OnDone | None = None,
    ) -> None:
        unknown = [fmt for fmt in formats if fmt not in FORMATS]
        if unknown:
            raise ParameterError(
                f"Форматы не поддерживаются: {', '.join(unknown)}; допустимые: "
                + ", ".join(FORMATS)
            )
        if summary is not None and Path(summary).suffix.lower().lstrip(".") not in SUMMARY_FORMATS:
            raise ParameterError(f"Общий файл должен быть CSV или JSONL: {summary}")
        if workers < 1:
            raise ParameterError(f"Число потоков должно быть положительным: {workers}")
        self.transcriber = transcriber
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.formats = list(formats)
        self.summary = None if summary is None else Path(summary)
        self.overwrite = overwrite
        self.delete_source = delete_source
        self.workers = workers
        self.log = log
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
            Ошибка одного файла не останавливает остальные: она пишется в лог и в отчет

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
            else:
                pending.append(job)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._process, job): job for job in pending}
            try:
                for number, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    prefix = f"[{number}/{len(pending)}] {job.source}"
                    try:
                        transcript, seconds = future.result()
                    except Exception as error:
                        report.failed[job.source] = str(error)
                        self.log(f"{prefix}: ошибка - {error}")
                    else:
                        report.done.append(job.source)
                        if self.on_done is not None:
                            self.on_done(job, transcript)
                        self.log(
                            f"{prefix}: фрагментов {len(transcript.segments)}, "
                            f"{transcript.duration:.1f} с записи за {seconds:.1f} с"
                        )
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        return report

    def _process(self, job: Job) -> tuple[Transcript, float]:
        started = time.perf_counter()
        transcript = self.transcriber.transcribe(job.source)
        seconds = time.perf_counter() - started
        for path, fmt in zip(self.outputs(job), self.formats, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(dumps(transcript, fmt))
        if self.summary is not None:
            self._append_summary(transcript)
        if self.delete_source:
            job.source.unlink()
        return transcript, seconds

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
) -> Report:
    """
    Мониторинг каталогов: распознавание новых файлов

    Описание:
        Каталоги просматриваются каждые interval секунд; файл берется в работу, когда
        не менялся min_age секунд. Файл с ошибкой не повторяется, пока не изменится

    Аргументы:
        batch (Batch): Распознавание с записью результатов
        directories (list[str]): Каталоги
        recursive (bool): Просматривать и вложенные каталоги
        interval (float): Пауза между просмотрами, с
        min_age (float): Сколько секунд файл не должен меняться
        rounds (int): Число просмотров; None - до прерывания
        sleep (Callable[[float], None]): Функция паузы
        clock (Callable[[], float]): Текущее время, с

    Вывод:
        Report: Итог всех просмотров
    """
    total = Report()
    attempts: dict[Path, tuple[float, int]] = {}
    count = 0
    while rounds is None or count < rounds:
        count += 1
        now = clock()
        ready = []
        for job in find_jobs(directories, recursive):
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
        _merge(total, report)
        if rounds is None or count < rounds:
            sleep(interval)
    return total


def _merge(total: Report, report: Report) -> None:
    total.done += report.done
    total.skipped += report.skipped
    total.failed.update(report.failed)


def summary_line(report: Report) -> str:
    """
    Итоговая строка отчета

    Аргументы:
        report (Report): Отчет

    Вывод:
        str: Сколько файлов распознано, пропущено и с ошибкой
    """
    total = len(report.done) + len(report.skipped) + len(report.failed)
    return (
        f"Готово: распознано {len(report.done)} из {total}, пропущено {len(report.skipped)}, "
        f"с ошибкой {len(report.failed)}"
    )
