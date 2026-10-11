import queue
import re
import secrets
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, Literal

from ..batch import Log, ignore
from ..exceptions import ParameterError, SttRuError
from ..pipeline import Transcriber
from ..result import Transcript

Status = Literal["queued", "running", "done", "failed"]
PEAKS = 2000
_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,10}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class Job:
    """
    Задача распознавания

    Атрибуты:
        id (str): Случайный идентификатор, его нельзя подобрать
        filename (str): Имя загруженного файла без каталогов
        path (Path): Файл на диске под случайным именем
        size (int): Размер файла, байт
        created (float): Время создания, с от начала эпохи
        status (str): queued - в очереди, running - распознается, done - готово,
            failed - ошибка
        started (float): Время начала распознавания
        finished (float): Время окончания
        error (str): Текст ошибки
        duration (float): Длительность записи, с; известна после чтения файла
        channels (int): Число каналов файла
        peaks (list[list[float]]): Пики формы волны по каналам (см. Audio.peaks)
        transcript (Transcript): Результат распознавания
    """

    id: str
    filename: str
    path: Path
    size: int
    created: float
    status: Status = "queued"
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    duration: float | None = None
    channels: int | None = None
    peaks: list[list[float]] | None = None
    transcript: Transcript | None = None
    deleted: bool = False


class QueueFullError(SttRuError):
    """В очереди нет места"""


class JobQueue:
    """
    Очередь задач распознавания в процессе сервиса

    Описание:
        Модель загружена один раз; задачи распознают workers потоков по порядку
        поступления. Файлы задач лежат во временном каталоге внутри directory и удаляются
        вместе с задачей: по запросу, через keep секунд после окончания или при close.
        Задачи хранятся в памяти и не переживают перезапуск сервиса

    Аргументы:
        transcriber (Transcriber): Распознаватель
        directory (str): Каталог для временного каталога файлов; None - системный
        workers (int): Сколько задач распознавать одновременно
        max_queue (int): Сколько задач может ждать в очереди
        max_duration (float): Наибольшая длительность записи, с; None - без ограничения
        keep (float): Сколько секунд хранить задачу после окончания
        log (Callable[[str], None]): Ход работы
        clock (Callable[[], float]): Текущее время, с от начала эпохи

    Исключения:
        ParameterError: Если workers, max_queue, max_duration или keep не положительны
    """

    def __init__(
        self,
        transcriber: Transcriber,
        directory: str | Path | None = None,
        *,
        workers: int = 1,
        max_queue: int = 100,
        max_duration: float | None = None,
        keep: float = 86400.0,
        log: Log = ignore,
        clock: Callable[[], float] = time.time,
    ) -> None:
        for name, value in (("workers", workers), ("max_queue", max_queue), ("keep", keep)):
            if value <= 0:
                raise ParameterError(f"{name} должен быть положительным: {value}")
        if max_duration is not None and max_duration <= 0:
            raise ParameterError(f"max_duration должен быть положительным: {max_duration}")
        self.transcriber = transcriber
        self.max_queue = max_queue
        self.max_duration = max_duration
        self.keep = keep
        self.log = log
        self.clock = clock
        self.directory = Path(tempfile.mkdtemp(prefix="stt-ru-", dir=directory))
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pending: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._threads = [
            threading.Thread(target=self._work, name=f"stt-ru-worker-{number}", daemon=True)
            for number in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, filename: str, content: BinaryIO) -> Job:
        """
        Постановка файла в очередь

        Аргументы:
            filename (str): Имя файла у пользователя; каталоги отбрасываются
            content (BinaryIO): Содержимое файла, читается до конца

        Вывод:
            Job: Задача в очереди

        Исключения:
            QueueFullError: Если в очереди max_queue задач
        """
        with self._lock:
            if self._waiting() >= self.max_queue:
                raise QueueFullError(f"Очередь заполнена: задач в ней {self.max_queue}")
        name = _CONTROL.sub("", PureWindowsPath(filename).name).strip()[-200:] or "audio"
        job_id = secrets.token_urlsafe(16)
        folder = self.directory / job_id
        folder.mkdir()
        suffix = _SUFFIX.search(name)
        path = folder / ("source" + (suffix.group().lower() if suffix else ""))
        try:
            with path.open("wb") as file:
                shutil.copyfileobj(content, file, 1 << 20)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        job = Job(job_id, name, path, path.stat().st_size, self.clock())
        self.log(f"[{job_id}] {name}: в очереди")
        with self._lock:
            self._jobs[job_id] = job
        self._pending.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        """
        Задача по идентификатору

        Аргументы:
            job_id (str): Идентификатор

        Вывод:
            Job: Копия задачи на момент вызова; None - нет такой задачи или она удалена
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None or job.deleted else replace(job)

    def position(self, job_id: str) -> int | None:
        """
        Место в очереди

        Аргументы:
            job_id (str): Идентификатор задачи

        Вывод:
            int: Сколько задач в очереди перед ней; None - задача не в очереди
        """
        with self._lock:
            ahead = 0
            for job in self._jobs.values():
                if job.id == job_id:
                    return ahead if job.status == "queued" and not job.deleted else None
                ahead += job.status == "queued" and not job.deleted
            return None

    def expires(self, job: Job) -> float | None:
        """
        Когда задача будет удалена

        Аргументы:
            job (Job): Задача

        Вывод:
            float: Время удаления, с от начала эпохи; None - задача не закончена
        """
        return None if job.finished is None else job.finished + self.keep

    def delete(self, job_id: str) -> bool:
        """
        Удаление задачи и ее файлов

        Описание:
            Задача в очереди не распознается; распознаваемая задача удаляется сразу
            после окончания

        Аргументы:
            job_id (str): Идентификатор

        Вывод:
            bool: Задача была
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.deleted:
                return False
            job.deleted = True
            if job.status == "running":
                return True
            del self._jobs[job_id]
        shutil.rmtree(job.path.parent, ignore_errors=True)
        self.log(f"[{job_id}] {job.filename}: удалена")
        return True

    def cleanup(self) -> int:
        """
        Удаление задач, которые закончились больше keep секунд назад

        Вывод:
            int: Сколько задач удалено
        """
        now = self.clock()
        with self._lock:
            expired = [
                job.id
                for job in self._jobs.values()
                if job.finished is not None and job.finished + self.keep <= now
            ]
        return sum(self.delete(job_id) for job_id in expired)

    def close(self) -> None:
        """Остановка потоков и удаление файлов всех задач"""
        for _ in self._threads:
            self._pending.put(None)
        with self._lock:
            for job in self._jobs.values():
                job.deleted = True
            self._jobs.clear()
        # В Windows файл, который еще читает поток распознавания, не удаляется
        shutil.rmtree(self.directory, ignore_errors=True)

    def _waiting(self) -> int:
        return sum(1 for job in self._jobs.values() if job.status == "queued" and not job.deleted)

    def _work(self) -> None:
        while (job_id := self._pending.get()) is not None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.deleted:
                    continue
                job.started = self.clock()
                job.status = "running"
            self._run(job)

    def _run(self, job: Job) -> None:
        try:
            audio = self.transcriber.load(job.path, max_duration=self.max_duration)
            peaks = [[round(float(peak), 3) for peak in row] for row in audio.peaks(PEAKS)]
            with self._lock:
                job.duration, job.channels, job.peaks = audio.duration, audio.num_channels, peaks
            transcript = self.transcriber.transcribe_audio(audio, source=job.filename)
        except Exception as error:
            with self._lock:
                job.error, job.finished = str(error) or type(error).__name__, self.clock()
                job.status = "failed"
            self.log(f"[{job.id}] {job.filename}: ошибка - {job.error}")
        else:
            with self._lock:
                job.transcript, job.finished = transcript, self.clock()
                job.status = "done"
            seconds = job.finished - (job.started or job.finished)
            self.log(
                f"[{job.id}] {job.filename}: фрагментов {len(transcript.segments)}, "
                f"{transcript.duration:.1f} с записи за {seconds:.1f} с"
            )
        with self._lock:
            removed = job.deleted and self._jobs.pop(job.id, None) is not None
        if removed:
            shutil.rmtree(job.path.parent, ignore_errors=True)
