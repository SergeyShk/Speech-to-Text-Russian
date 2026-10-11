import asyncio
import mimetypes
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Annotated
from urllib.parse import quote, unquote

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from ..export import FORMATS, dumps
from .worker import Job, JobQueue, QueueFullError, Status

STATIC = Path(__file__).parent / "static"
TOKEN_COOKIE = "stt_ru_token"
UPLOAD_PATH = "/v1/transcriptions"
MISSING = "Задачи нет: она удалена или срок хранения вышел"
# Заголовки и границы частей multipart сверх самого файла
UPLOAD_OVERHEAD = 1 << 16
MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "ass": "text/x-ssa; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
}
PAGE_POLICY = (
    "default-src 'self'; img-src 'self' data:; media-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


class JobInfo(BaseModel):
    """Состояние задачи"""

    id: str = Field(description="Идентификатор задачи")
    status: Status = Field(
        description="queued - в очереди, running - распознается, done - готово, failed - ошибка"
    )
    filename: str = Field(description="Имя загруженного файла")
    size: int = Field(description="Размер файла, байт")
    created: datetime = Field(description="Время создания")
    started: datetime | None = Field(None, description="Начало распознавания")
    finished: datetime | None = Field(None, description="Окончание")
    expires: datetime | None = Field(None, description="Когда задача и файлы будут удалены")
    position: int | None = Field(None, description="Сколько задач в очереди перед этой")
    duration: float | None = Field(None, description="Длительность записи, с")
    channels: int | None = Field(None, description="Число каналов файла")
    segments: int | None = Field(None, description="Число распознанных фрагментов")
    error: str | None = Field(None, description="Текст ошибки")


class ServiceInfo(BaseModel):
    """Сведения о сервисе"""

    version: str = Field(description="Версия stt-ru")
    model: str = Field(description="Модель распознавания")
    auth: bool = Field(description="Нужен токен доступа")
    authorized: bool = Field(description="Запрос с верным токеном или токен не нужен")
    max_size: int = Field(description="Наибольший размер файла, байт")
    max_duration: float | None = Field(description="Наибольшая длительность записи, с")
    keep: float = Field(description="Сколько секунд задача хранится после окончания")
    formats: list[str] = Field(description="Форматы результата")


class Waveform(BaseModel):
    """Форма волны для отображения"""

    duration: float = Field(description="Длительность записи, с")
    peaks: list[list[float]] = Field(description="Пики амплитуды по каналам, от 0 до 1")


@dataclass(frozen=True)
class _Settings:
    jobs: JobQueue
    token: str | None
    max_size: int


def create_app(jobs: JobQueue, *, token: str | None = None, max_size: int = 500 << 20) -> FastAPI:
    """
    Веб-приложение сервиса: страница и REST API

    Описание:
        Страница - на /, REST API - на /v1, описание API - на /docs. Если задан token,
        запросы к задачам требуют заголовок Authorization: Bearer <token> или cookie
        stt_ru_token (ее ставит страница). Файл больше max_size отклоняется во время
        загрузки. Не реже раза в минуту удаляются задачи, срок хранения которых вышел;
        при остановке приложения очередь закрывается

    Аргументы:
        jobs (JobQueue): Очередь задач
        token (str): Токен доступа; None или пустая строка - доступ без токена
        max_size (int): Наибольший размер файла, байт

    Вывод:
        FastAPI: Приложение ASGI
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(_cleanup(jobs))
        try:
            yield
        finally:
            task.cancel()
            jobs.close()

    app = FastAPI(
        title="stt-ru",
        version=__version__,
        summary="Распознавание русской речи",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = _Settings(jobs, token or None, max_size)
    app.add_middleware(_Guard, max_body=max_size + UPLOAD_OVERHEAD)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.include_router(router)
    return app


def _settings(request: Request) -> _Settings:
    settings: _Settings = request.app.state.settings
    return settings


Settings = Annotated[_Settings, Depends(_settings)]


def _authorized(request: Request, settings: _Settings) -> bool:
    if settings.token is None:
        return True
    header = request.headers.get("authorization", "")
    if header[:7].lower() == "bearer ":
        # Starlette декодирует заголовки как latin-1, а токен может быть не ASCII
        given = header[7:].strip().encode("latin-1")
    else:
        given = unquote(request.cookies.get(TOKEN_COOKIE, "")).encode()
    return secrets.compare_digest(given, settings.token.encode())


def _authorize(request: Request, settings: Settings) -> None:
    if not _authorized(request, settings):
        raise HTTPException(401, "Нужен токен доступа", {"WWW-Authenticate": "Bearer"})


def _find(settings: _Settings, job_id: str) -> Job:
    job = settings.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, MISSING)
    return job


def _info(jobs: JobQueue, job: Job) -> JobInfo:
    return JobInfo(
        id=job.id,
        status=job.status,
        filename=job.filename,
        size=job.size,
        created=datetime.fromtimestamp(job.created, UTC),
        started=_time(job.started),
        finished=_time(job.finished),
        expires=_time(jobs.expires(job)),
        position=jobs.position(job.id),
        duration=job.duration,
        channels=job.channels,
        segments=None if job.transcript is None else len(job.transcript.segments),
        error=job.error,
    )


router = APIRouter()
protected = [Depends(_authorize)]


@router.get("/", include_in_schema=False)
def page() -> FileResponse:
    headers = {"Content-Security-Policy": PAGE_POLICY, "Cache-Control": "no-cache"}
    return FileResponse(STATIC / "index.html", headers=headers)


@router.get("/healthz", summary="Проверка работы")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/info", summary="Сведения о сервисе")
def service_info(request: Request, settings: Settings) -> ServiceInfo:
    return ServiceInfo(
        version=__version__,
        model=settings.jobs.transcriber.recognizer.model,
        auth=settings.token is not None,
        authorized=_authorized(request, settings),
        max_size=settings.max_size,
        max_duration=settings.jobs.max_duration,
        keep=settings.jobs.keep,
        formats=list(FORMATS),
    )


@router.post(
    UPLOAD_PATH,
    status_code=202,
    summary="Поставить файл в очередь",
    dependencies=protected,
    responses={
        413: {"description": "Файл больше наибольшего размера"},
        503: {"description": "Очередь заполнена"},
    },
)
def create_job(file: UploadFile, response: Response, settings: Settings) -> JobInfo:
    if file.size is not None and file.size > settings.max_size:
        raise HTTPException(413, _too_large(settings.max_size))
    try:
        job = settings.jobs.submit(file.filename or "audio", file.file)
    except QueueFullError as error:
        raise HTTPException(503, str(error), {"Retry-After": "60"}) from error
    response.headers["Location"] = f"/v1/jobs/{job.id}"
    return _info(settings.jobs, job)


@router.get("/v1/jobs/{job_id}", summary="Состояние задачи", dependencies=protected)
def get_job(job_id: str, settings: Settings) -> JobInfo:
    return _info(settings.jobs, _find(settings, job_id))


@router.get(
    "/v1/jobs/{job_id}/result",
    summary="Результат распознавания",
    dependencies=protected,
    response_class=Response,
    responses={409: {"description": "Задача не готова или закончилась ошибкой"}},
)
def get_result(
    job_id: str,
    settings: Settings,
    fmt: Annotated[
        str, Query(alias="format", description="Формат: " + ", ".join(FORMATS))
    ] = "json",
) -> Response:
    job = _find(settings, job_id)
    if fmt not in FORMATS:
        raise HTTPException(400, f"Неизвестный формат {fmt}: допустимы {', '.join(FORMATS)}")
    if job.status == "failed":
        raise HTTPException(409, f"Распознавание не удалось: {job.error}")
    if job.transcript is None:
        raise HTTPException(409, "Задача еще не готова")
    name = f"{PurePath(job.filename).stem or 'audio'}.{fmt}"
    return Response(
        dumps(job.transcript, fmt),
        media_type=MEDIA_TYPES[fmt],
        headers={"Content-Disposition": _disposition("attachment", name)},
    )


@router.get(
    "/v1/jobs/{job_id}/waveform",
    summary="Форма волны",
    dependencies=protected,
    responses={409: {"description": "Файл еще не прочитан"}},
)
def get_waveform(job_id: str, settings: Settings) -> Waveform:
    job = _find(settings, job_id)
    if job.peaks is None or job.duration is None:
        raise HTTPException(409, "Файл еще не прочитан")
    return Waveform(duration=job.duration, peaks=job.peaks)


@router.get(
    "/v1/jobs/{job_id}/audio",
    summary="Загруженный файл",
    dependencies=protected,
    response_class=FileResponse,
)
def get_audio(job_id: str, settings: Settings) -> FileResponse:
    job = _find(settings, job_id)
    media_type = mimetypes.guess_type(job.filename)[0] or "application/octet-stream"
    return FileResponse(
        job.path,
        media_type=media_type,
        headers={"Content-Disposition": _disposition("inline", job.filename)},
    )


@router.delete(
    "/v1/jobs/{job_id}", status_code=204, summary="Удалить задачу", dependencies=protected
)
def delete_job(job_id: str, settings: Settings) -> Response:
    if not settings.jobs.delete(job_id):
        raise HTTPException(404, MISSING)
    return Response(status_code=204)


async def _cleanup(jobs: JobQueue) -> None:
    while True:
        await asyncio.sleep(min(60.0, jobs.keep))
        await asyncio.to_thread(jobs.cleanup)


def _time(value: float | None) -> datetime | None:
    return None if value is None else datetime.fromtimestamp(value, UTC)


def _too_large(limit: int) -> str:
    return f"Файл больше {limit / (1 << 20):g} МБ"


def _disposition(kind: str, name: str) -> str:
    fallback = name.encode("ascii", "replace").decode().replace('"', "_").replace("?", "_")
    return f"{kind}; filename=\"{fallback}\"; filename*=UTF-8''{quote(name)}"


class _Guard:
    """Заголовки безопасности и предел тела запроса загрузки"""

    def __init__(self, app: ASGIApp, max_body: int) -> None:
        self.app = app
        self.max_body = max_body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def secure(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                ]
                message = {**message, "headers": headers}
            await send(message)

        if scope["method"] != "POST" or scope["path"] != UPLOAD_PATH:
            await self.app(scope, receive, secure)
            return
        length = dict(scope["headers"]).get(b"content-length")
        limit = self.max_body - UPLOAD_OVERHEAD
        if length is not None and length.isdigit() and int(length) > self.max_body:
            await JSONResponse({"detail": _too_large(limit)}, 413)(scope, receive, secure)
            return
        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_body:
                raise HTTPException(413, _too_large(limit))
            return message

        await self.app(scope, counted, secure)
