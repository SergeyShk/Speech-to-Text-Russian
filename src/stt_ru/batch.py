import dataclasses
import math
import os
import struct
import sys
import threading
import time
from collections.abc import Callable, Collection, Sequence
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
Signature = tuple[float, int]
Result = tuple[Transcript, float, Signature | None]


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
        unsaved (list[Path]): Файлы из failed, которые распознаны, но результаты
            не записались: причина может быть временной
    """

    done: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: dict[Path, str] = field(default_factory=dict)
    unsaved: list[Path] = field(default_factory=list)


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
    names: bool = True,
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
        strict (bool): Отсутствующий или недоступный путь из аргументов и совпадение имен
            результатов - ошибка; False - они пропускаются с сообщением в log
        names (bool): Проверять совпадение имен результатов; False - когда результаты
            не сохраняются в каталог
        log (Callable[[str], None]): Сообщения о недоступных вложенных каталогах, а при
            strict=False - и о пропущенных путях и файлах

    Вывод:
        list[Job]: Файлы в порядке аргументов, внутри каталога - по пути

    Исключения:
        FileNotFoundError: Если пути нет и strict=True
        PermissionError: Если путь из аргументов недоступен и strict=True
        ParameterError: Если у двух файлов совпадают имена результатов и strict=True
    """
    jobs: list[Job] = []
    for item in map(Path, paths):
        try:
            if item.is_dir():
                jobs += _directory_jobs(item, recursive, log)
            elif item.is_file():
                jobs.append(Job(item, Path(item.stem), item.name))
            else:
                raise FileNotFoundError(f"Файл или каталог не найден: {item}")
        except OSError as error:
            if strict:
                raise
            log(str(error))
    unique: dict[Path, Job] = {}
    for job in jobs:
        unique.setdefault(job.source.resolve(), job)
    if not names:
        return list(unique.values())
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


def _directory_jobs(directory: Path, recursive: bool, log: Log) -> list[Job]:
    jobs = []
    denied: list[OSError] = []
    # os.walk, в отличие от glob, сообщает о каталогах, которые не удалось прочитать
    for root, folders, files in os.walk(directory, onerror=denied.append):
        folders[:] = [name for name in folders if not name.startswith(".")] if recursive else []
        for name in files:
            file = Path(root, name)
            if name.startswith(".") or file.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            if file.is_file():
                relative = file.relative_to(directory)
                jobs.append(Job(file, relative.with_suffix(""), relative.as_posix()))
    for error in denied:
        # Недоступен сам каталог из аргументов - ошибка; вложенный пропускается
        if Path(error.filename) == directory:
            raise PermissionError(f"Нет доступа к каталогу: {directory}")
        log(f"Нет доступа к каталогу: {error.filename}")
    return sorted(jobs, key=lambda job: job.source.relative_to(directory).parts)


def check_options(
    formats: Sequence[str],
    summary: str | os.PathLike[str] | None = None,
    workers: int = 1,
    output_dir: str | os.PathLike[str] | None = None,
) -> None:
    """
    Проверка параметров пакетной обработки

    Аргументы:
        formats (list[str]): Форматы результатов
        summary (str): Общий файл
        workers (int): Число потоков
        output_dir (str): Каталог результатов

    Исключения:
        ParameterError: Если формат не поддерживается, для каталога результатов нет
            форматов, у общего файла не то расширение, каталог результатов или общий
            файл нельзя записать или число потоков не положительно
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
    if output_dir is not None:
        if not formats:
            raise ParameterError("Для каталога результатов нужен хотя бы один формат")
        check_target(output_dir, directory=True)
    if summary is not None:
        check_target(summary, directory=False, append=True)


def check_target(path: str | os.PathLike[str], *, directory: bool, append: bool = False) -> None:
    """
    Проверка, что файл или каталог результата можно записать

    Описание:
        Путь должен быть каталогом (directory=True) или не быть каталогом, а ближайший
        существующий каталог на пути - доступен на запись: недостающие каталоги
        создаются при записи. Файл, который дописывается (append=True), если он есть,
        должен быть доступен на запись сам

    Аргументы:
        path (str): Путь к файлу или каталогу
        directory (bool): Путь - каталог
        append (bool): Файл дописывается, а не заменяется

    Исключения:
        ParameterError: Если путь пуст, не того вида или недоступен на запись
    """
    if not os.fspath(path):
        raise ParameterError("Путь результата не задан")
    target = Path(path)
    if target.exists() and target.is_dir() != directory:
        raise ParameterError(
            f"{target} - {'файл' if directory else 'каталог'}, а нужен "
            f"{'каталог' if directory else 'файл'}"
        )
    if append and target.is_file():
        if not os.access(target, os.W_OK):
            raise ParameterError(f"Нет права записи в файл {target}")
        return
    folder = target if directory else target.parent
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    if not folder.is_dir():
        raise ParameterError(f"{folder} - файл, а нужен каталог")
    if not os.access(folder, os.W_OK | os.X_OK):
        raise ParameterError(f"Нет права записи в каталог {folder}")


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
        все результаты, пропускается и не удаляется: это может быть другая запись
        с тем же именем

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
        check_options(formats, summary, workers, output_dir)
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
        pending = [job for job in jobs if not self._skip(job, report)]
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

    def _skip(self, job: Job, report: Report, force: Collection[Path] = ()) -> bool:
        outputs = self.outputs(job)
        if self.overwrite or job.source in force or not outputs:
            return False
        if not all(path.exists() for path in outputs):
            return False
        report.skipped.append(job.source)
        kept = ", исходник оставлен" if self.delete_source else ""
        self.log(f"{job.source}: пропущен, результаты уже есть{kept}")
        return True

    def _transcribe(self, job: Job) -> Result:
        started = time.perf_counter()
        before = _signature(job.source)
        transcript = self.transcriber.transcribe(job.source)
        return transcript, time.perf_counter() - started, before

    def _finish(self, job: Job, future: "Future[Result]", report: Report, prefix: str) -> None:
        try:
            transcript, seconds, before = future.result()
        except Exception as error:
            self._fail(job, report, prefix, error)
            return
        # Файл дописали, пока он распознавался: результат неполный, исходник нужен целиком
        after = _signature(job.source)
        if self.delete_source and after is not None and after != before:
            self._fail(job, report, prefix, "файл изменился во время распознавания")
            return
        try:
            self._save(job, transcript)
        except Exception as error:
            report.unsaved.append(job.source)
            self._fail(job, report, prefix, error)
            return
        if self.on_done is not None:
            self.on_done(job, transcript)
        if self.delete_source and not self._delete(job, report):
            return
        report.done.append(job.source)
        self.log(
            f"{prefix} {job.source}: фрагментов {len(transcript.segments)}, "
            f"{transcript.duration:.1f} с записи за {seconds:.1f} с"
        )

    def _fail(self, job: Job, report: Report, prefix: str, error: object) -> None:
        report.failed[job.source] = str(error)
        self.errors(f"{prefix} {job.source}: ошибка - {error}")

    def _save(self, job: Job, transcript: Transcript) -> None:
        if self.summary is not None:
            self._append_summary(dataclasses.replace(transcript, source=job.label))
        for path, fmt in zip(self.outputs(job), self.formats, strict=True):
            write_atomic(path, dumps(transcript, fmt))

    def _delete(self, job: Job, report: Report) -> bool:
        try:
            job.source.unlink(missing_ok=True)
        except OSError as error:
            report.failed[job.source] = f"результаты сохранены, исходник не удален: {error}"
            self.errors(f"{job.source}: результаты сохранены, исходник не удален - {error}")
            return False
        return True

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
        не менялся min_age секунд, а в WAV - когда данных столько, сколько указано
        в заголовке (о недописанном WAV пишется сообщение). Новые файлы ставятся
        в очередь, не дожидаясь файлов прошлых
        просмотров; одновременно распознается batch.workers файлов, результаты
        сохраняются в начале следующего просмотра. Файл, который изменился после
        распознавания, распознается заново с перезаписью результатов (строки общего
        файла дописываются еще раз). Файл с ошибкой распознавания не повторяется, пока
        не изменится; файл, результаты которого не записались, повторяется в следующем
        просмотре. Пропавший или недоступный каталог и файлы с совпадающими именами
        результатов не останавливают мониторинг: сообщение о них пишется в batch.errors,
        когда случай появляется. Попытки файлов из недоступного каталога помнятся
        до его возвращения. При прерывании (Ctrl+C) очередь отменяется, а файлы в работе
        дораспознаются и сохраняются

    Аргументы:
        batch (Batch): Распознавание с записью результатов
        directories (list[str]): Каталоги
        recursive (bool): Просматривать и вложенные каталоги
        interval (float): Пауза между просмотрами, с, больше 0
        min_age (float): Сколько секунд файл не должен меняться, не меньше 0
        rounds (int): Число просмотров; None - до прерывания. После последнего
            просмотра файлы в работе дораспознаются
        sleep (Callable[[float], None]): Функция паузы
        clock (Callable[[], float]): Текущее время, с

    Вывод:
        WatchReport: Итог всех просмотров

    Исключения:
        ParameterError: Если interval или min_age не проходят check_watch
    """
    check_watch(interval, min_age)
    state = _Watch(batch)
    executor = ThreadPoolExecutor(max_workers=batch.workers)
    count = 0
    try:
        while rounds is None or count < rounds:
            count += 1
            state.collect()
            for job in state.scan(directories, recursive, min_age, clock()):
                state.running[executor.submit(batch._transcribe, job)] = job
            if rounds is None or count < rounds:
                sleep(interval)
        wait(state.running)
        state.collect()
    except KeyboardInterrupt:
        busy = [future for future in state.running if not future.cancel()]
        state.running = {future: state.running[future] for future in busy}
        if busy:
            batch.log(
                f"Прерывание: дожидаюсь файлов в работе ({len(busy)}), "
                "Ctrl+C еще раз - выйти сразу"
            )
            wait(busy)
            state.collect()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return state.total


class _Watch:
    def __init__(self, batch: Batch) -> None:
        self.batch = batch
        self.total = WatchReport()
        # Сигнатура файла при последней попытке: изменившийся файл берется снова
        self.attempts: dict[Path, Signature] = {}
        # Файлы, результаты которых при повторе надо перезаписать
        self.forced: set[Path] = set()
        self.running: dict[Future[Result], Job] = {}
        self.active: set[str] = set()
        self.finished = 0

    def collect(self) -> None:
        for future in [future for future in self.running if future.done()]:
            job = self.running.pop(future)
            self.finished += 1
            report = Report()
            self.batch._finish(job, future, report, f"[{self.finished}]")
            self.total.done += len(report.done)
            for path in report.done:
                self.total.failed.pop(path, None)
            self.total.failed.update(report.failed)
            for path in report.unsaved:
                # Повтор с перезаписью: на диске могут быть результаты прошлой версии файла
                self.attempts.pop(path, None)
                self.forced.add(path)

    def scan(
        self,
        directories: Sequence[str | os.PathLike[str]],
        recursive: bool,
        min_age: float,
        now: float,
    ) -> list[Job]:
        messages: list[str] = []
        try:
            jobs = find_jobs(directories, recursive, strict=False, log=messages.append)
        except OSError as error:
            messages.append(f"Не удалось просмотреть каталоги: {error}")
            jobs = []
        roots = [Path(directory) for directory in directories if Path(directory).is_dir()]
        present = {job.source for job in jobs}
        self.attempts = {
            path: value for path, value in self.attempts.items() if not _gone(path, present, roots)
        }
        self.total.failed = {
            path: text
            for path, text in self.total.failed.items()
            if not _gone(path, present, roots)
        }
        busy = set(self.running.values())
        ready = []
        for job in jobs:
            signature = _signature(job.source)
            if job in busy or signature is None or now - signature[0] < min_age:
                continue
            previous = self.attempts.get(job.source)
            if previous == signature:
                continue
            if not _complete(job.source):
                messages.append(
                    f"{job.source}: в WAV меньше данных, чем в заголовке, ждет дописывания"
                )
                continue
            self.attempts[job.source] = signature
            # Файл изменился после попытки: прежние результаты могли быть по его части
            forced = previous is not None or job.source in self.forced
            self.forced.discard(job.source)
            if self.batch._skip(job, Report(), {job.source} if forced else set()):
                self.total.skipped += 1
            else:
                ready.append(job)
        for message in messages:
            if message not in self.active:
                self.batch.errors(message)
        self.active = set(messages)
        return ready


def _signature(path: Path) -> Signature | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime, stat.st_size


def _complete(path: Path) -> bool:
    # WAV копируют от начала: в заголовке уже полный размер данных, а самих данных меньше
    if path.suffix.lower() != ".wav":
        return True
    try:
        with path.open("rb") as file:
            head = file.read(1 << 16)
            size = os.fstat(file.fileno()).st_size
    except OSError:
        return True
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return True
    offset = 12
    while offset + 8 <= len(head):
        chunk = head[offset : offset + 4]
        (length,) = struct.unpack("<I", head[offset + 4 : offset + 8])
        if chunk == b"data":
            # 0 и 0xFFFFFFFF пишут программы, которые не знают длину записи заранее
            return length in (0, 0xFFFFFFFF) or offset + 8 + length <= size
        offset += 8 + length + (length & 1)
    return True


def _gone(path: Path, present: set[Path], roots: Sequence[Path]) -> bool:
    if path in present or not any(path.is_relative_to(root) for root in roots):
        return False
    # Файл под недоступным каталогом не пропал: его попытка нужна, когда доступ вернется
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def check_watch(interval: float, min_age: float) -> None:
    """
    Проверка параметров мониторинга

    Аргументы:
        interval (float): Пауза между просмотрами, с
        min_age (float): Сколько секунд файл не должен меняться

    Исключения:
        ParameterError: Если interval не конечное положительное число или min_age
            не конечное неотрицательное
    """
    if not (math.isfinite(interval) and interval > 0):
        raise ParameterError(f"Интервал просмотра должен быть положительным числом: {interval}")
    if not (math.isfinite(min_age) and min_age >= 0):
        raise ParameterError(f"Время без изменений должно быть неотрицательным числом: {min_age}")


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
