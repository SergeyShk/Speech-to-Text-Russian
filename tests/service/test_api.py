import json
import time
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

import stt_ru
from stt_ru import Transcriber
from stt_ru.service import JobQueue, api, create_app

from .conftest import wait, wav_bytes

CALL = Path(__file__).parent.parent / "data" / "call.wav"


def make_client(jobs, **options):
    return TestClient(create_app(jobs, **options))


@pytest.fixture
def client(jobs):
    with make_client(jobs) as client:
        yield client


def upload(client, name="call.wav", content=None, **kwargs):
    files = {"file": (name, wav_bytes() if content is None else content, "audio/wav")}
    return client.post("/v1/transcriptions", files=files, **kwargs)


def finished(client, jobs, name="call.wav"):
    response = upload(client, name)
    assert response.status_code == 202
    return wait(jobs, response.json()["id"])


class TestPage:
    def test_index(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        policy = response.headers["content-security-policy"]
        assert "default-src 'self'" in policy and "img-src 'self';" in policy
        assert "data:" not in policy
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert '<script src="/static/app.js"' in response.text
        assert response.text.count('src="/static/icon.svg"') == 1
        assert '<link rel="icon" href="/static/icon.svg"' in response.text

    @pytest.mark.parametrize("name", ["app.js", "style.css", "icon.svg"])
    def test_static(self, client, name):
        response = client.get(f"/static/{name}")
        assert response.status_code == 200
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-cache"
        again = client.get(f"/static/{name}", headers={"If-None-Match": response.headers["etag"]})
        assert again.status_code == 304

    def test_api_not_marked_static(self, client):
        assert "cache-control" not in client.get("/healthz").headers

    def test_health(self, client):
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_info(self, client, jobs):
        assert client.get("/v1/info").json() == {
            "version": stt_ru.__version__,
            "model": "fake",
            "auth": False,
            "authorized": True,
            "max_size": 500 << 20,
            "max_duration": None,
            "keep": 86400.0,
            "formats": ["txt", "json", "jsonl", "srt", "vtt", "ass", "csv"],
        }

    def test_openapi(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {
            "/healthz",
            "/v1/info",
            "/v1/transcriptions",
            "/v1/jobs/{job_id}",
            "/v1/jobs/{job_id}/result",
            "/v1/jobs/{job_id}/waveform",
            "/v1/jobs/{job_id}/audio",
        }


class TestJobs:
    def test_create(self, client, jobs):
        response = upload(client, "звонок.wav")
        assert response.status_code == 202
        job = response.json()
        assert response.headers["location"] == f"/v1/jobs/{job['id']}"
        assert job["filename"] == "звонок.wav" and job["size"] == len(wav_bytes())
        assert job["status"] in ("queued", "running", "done")
        wait(jobs, job["id"])
        state = client.get(f"/v1/jobs/{job['id']}").json()
        assert state["status"] == "done"
        assert (state["duration"], state["channels"], state["segments"]) == (1.0, 2, 2)
        assert state["position"] is None and state["error"] is None
        assert state["created"] <= state["started"] <= state["finished"] < state["expires"]

    def test_queued(self, client, jobs, transcriber):
        transcriber.gate.clear()
        running = upload(client).json()["id"]
        wait(jobs, running, ("running",))
        queued = upload(client).json()
        assert (queued["status"], queued["position"]) == ("queued", 0)
        assert client.get(f"/v1/jobs/{queued['id']}/waveform").status_code == 409
        response = client.get(f"/v1/jobs/{running}/result")
        assert (response.status_code, response.json()) == (409, {"detail": "Задача еще не готова"})
        waveform = client.get(f"/v1/jobs/{running}/waveform").json()
        assert waveform["duration"] == 1.0 and len(waveform["peaks"]) == 2

    def test_failed(self, client, jobs):
        job = finished(client, jobs, "broken.wav")
        state = client.get(f"/v1/jobs/{job.id}").json()
        assert (state["status"], state["error"]) == ("failed", "модель не приняла запись")
        response = client.get(f"/v1/jobs/{job.id}/result?format=srt")
        assert response.status_code == 409
        assert response.json()["detail"] == "Распознавание не удалось: модель не приняла запись"

    def test_missing(self, client):
        for path in ("", "/result", "/waveform", "/audio"):
            response = client.get(f"/v1/jobs/missing{path}")
            assert response.status_code == 404
            assert response.json()["detail"].startswith("Задачи нет")
        assert client.delete("/v1/jobs/missing").status_code == 404

    def test_delete(self, client, jobs):
        job = finished(client, jobs)
        assert client.delete(f"/v1/jobs/{job.id}").status_code == 204
        assert client.get(f"/v1/jobs/{job.id}").status_code == 404
        assert not job.path.exists()

    def test_queue_full(self, transcriber, tmp_path):
        transcriber.gate.clear()
        jobs = JobQueue(transcriber, tmp_path, max_queue=1)
        with make_client(jobs) as client:
            wait(jobs, upload(client).json()["id"], ("running",))
            assert upload(client).status_code == 202
            response = upload(client)
            transcriber.gate.set()
        assert response.status_code == 503
        assert response.headers["retry-after"] == "60"
        assert response.json()["detail"] == "Очередь заполнена: задач в ней 1"

    def test_missing_file_field(self, client):
        assert client.post("/v1/transcriptions", data={"name": "x"}).status_code == 422


class TestResult:
    @pytest.mark.parametrize(
        ("fmt", "media_type", "start"),
        [
            ("txt", "text/plain; charset=utf-8", "Канал 0: Алло.".encode()),
            ("srt", "application/x-subrip; charset=utf-8", b"1\n00:00:00,100 --> "),
            ("vtt", "text/vtt; charset=utf-8", b"WEBVTT"),
            ("ass", "text/x-ssa; charset=utf-8", b"[Script Info]"),
            ("csv", "text/csv; charset=utf-8", "﻿".encode()),
            ("jsonl", "application/x-ndjson", b'{"source": "'),
        ],
    )
    def test_formats(self, client, jobs, fmt, media_type, start):
        job = finished(client, jobs, "Звонок 1.wav")
        response = client.get(f"/v1/jobs/{job.id}/result", params={"format": fmt})
        assert response.status_code == 200
        assert response.headers["content-type"] == media_type
        assert response.content.startswith(start)
        assert response.headers["content-disposition"] == (
            f"attachment; filename=\"______ 1.{fmt}\"; filename*=UTF-8''{quote(f'Звонок 1.{fmt}')}"
        )

    def test_json_by_default(self, client, jobs):
        job = finished(client, jobs, "call")
        response = client.get(f"/v1/jobs/{job.id}/result")
        assert response.headers["content-disposition"].startswith(
            'attachment; filename="call.json"'
        )
        document = response.json()
        assert document["source"] == "call"
        assert [segment["speaker"] for segment in document["segments"]] == ["Канал 0", "Канал 1"]

    def test_unknown_format(self, client, jobs):
        job = finished(client, jobs)
        response = client.get(f"/v1/jobs/{job.id}/result?format=docx")
        assert response.status_code == 400
        assert response.json()["detail"].startswith("Неизвестный формат docx: допустимы txt")

    def test_quote_in_name(self):
        assert api._disposition("attachment", 'a"b?.json') == (
            "attachment; filename=\"a_b_.json\"; filename*=UTF-8''a%22b%3F.json"
        )


class TestAudio:
    def test_range(self, client, jobs):
        job = finished(client, jobs)
        response = client.get(f"/v1/jobs/{job.id}/audio", headers={"Range": "bytes=0-99"})
        assert response.status_code == 206
        assert response.content == wav_bytes()[:100]
        assert response.headers["content-type"] in ("audio/wav", "audio/x-wav")
        assert response.headers["content-disposition"].startswith('inline; filename="call.wav"')

    def test_unknown_type(self, client, jobs):
        job = finished(client, jobs, "запись")
        response = client.get(f"/v1/jobs/{job.id}/audio")
        assert response.headers["content-type"] == "application/octet-stream"


class TestLimits:
    def test_content_length(self, jobs):
        with make_client(jobs, max_size=1000) as client:
            response = upload(client, content=b"x" * 70_000)
            assert list(jobs.directory.iterdir()) == []
        assert response.status_code == 413
        assert response.json() == {"detail": "Файл больше 0.000953674 МБ"}

    def test_streamed_body(self, jobs):
        def body():
            yield b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
            for _ in range(100):
                yield b"x" * 1000

        with make_client(jobs, max_size=1000) as client:
            response = client.post(
                "/v1/transcriptions",
                content=body(),
                headers={"Content-Type": "multipart/form-data; boundary=x"},
            )
            assert list(jobs.directory.iterdir()) == []
        assert response.status_code == 413

    def test_file_size(self, jobs):
        with make_client(jobs, max_size=1000) as client:
            response = upload(client, content=b"x" * 1500)
        assert response.status_code == 413
        assert response.json()["detail"].startswith("Файл больше")

    def test_other_requests_not_limited(self, jobs):
        with make_client(jobs, max_size=10) as client:
            assert client.post("/v1/info", content=b"x" * 100).status_code == 405


class TestToken:
    @pytest.fixture
    def secured(self, jobs):
        with make_client(jobs, token="секрет;42") as client:
            yield client

    def test_required(self, secured, jobs):
        for method, path in [
            ("post", "/v1/transcriptions"),
            ("get", "/v1/jobs/x"),
            ("get", "/v1/jobs/x/result"),
            ("get", "/v1/jobs/x/waveform"),
            ("get", "/v1/jobs/x/audio"),
            ("delete", "/v1/jobs/x"),
        ]:
            response = secured.request(method, path)
            assert response.status_code == 401, path
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.json() == {"detail": "Нужен токен доступа"}
        assert list(jobs.directory.iterdir()) == []

    def test_public(self, secured):
        assert secured.get("/").status_code == 200
        assert secured.get("/healthz").status_code == 200
        info = secured.get("/v1/info").json()
        assert (info["auth"], info["authorized"]) == (True, False)

    def test_header(self, secured, jobs):
        headers = {"Authorization": "Bearer секрет;42".encode()}
        response = upload(secured, headers=headers)
        assert response.status_code == 202
        assert secured.get(f"/v1/jobs/{response.json()['id']}", headers=headers).status_code == 200
        assert secured.get("/v1/info", headers=headers).json()["authorized"]

    def test_cookie(self, secured):
        secured.cookies.set("stt_ru_token", quote("секрет;42"))
        assert upload(secured).status_code == 202
        assert secured.get("/v1/info").json()["authorized"]

    def test_empty_token_disables_auth(self, jobs):
        with make_client(jobs, token="") as client:
            assert client.get("/v1/info").json()["auth"] is False
            assert upload(client).status_code == 202

    @pytest.mark.parametrize("value", ["Bearer секрет", "Basic секрет;42", "секрет;42", "Bearer"])
    def test_wrong(self, secured, value):
        assert upload(secured, headers={"Authorization": value.encode()}).status_code == 401


def test_cleanup_task(transcriber, tmp_path):
    jobs = JobQueue(transcriber, tmp_path, keep=0.05)
    with make_client(jobs) as client:
        job = finished(client, jobs)
        deadline = time.monotonic() + 5
        while jobs.get(job.id) is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert client.get(f"/v1/jobs/{job.id}").status_code == 404
    assert not jobs.directory.exists()


def test_job_info_is_json(client, jobs):
    job = finished(client, jobs)
    assert json.loads(client.get(f"/v1/jobs/{job.id}").text)["id"] == job.id


@pytest.mark.network
def test_real_model(tmp_path):
    jobs = JobQueue(Transcriber(), tmp_path)
    with make_client(jobs) as client:
        response = client.post(
            "/v1/transcriptions", files={"file": ("call.wav", CALL.read_bytes(), "audio/wav")}
        )
        job = wait(jobs, response.json()["id"], timeout=120)
        assert job.status == "done", job.error
        srt = client.get(f"/v1/jobs/{job.id}/result", params={"format": "srt"}).text
        waveform = client.get(f"/v1/jobs/{job.id}/waveform").json()
    assert "Канал 1: Наталья, благодарю вас за ожидание." in srt
    assert len(waveform["peaks"]) == 2
