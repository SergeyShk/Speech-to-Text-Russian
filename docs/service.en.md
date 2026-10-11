# Web service

!!! info ""
    **stt-ru serve**, **stt_ru.service**

## Description

`stt-ru serve` runs a speech recognition web interface and REST API. The model is loaded once, files are transcribed in a queue inside the process, and the result can be downloaded in any [format](formats.md). The service is meant for one user or a small team. The `service` extra is required:

``` bash
pip install "stt-ru[service]"
stt-ru serve                                         # http://127.0.0.1:8000
STT_RU_TOKEN=my-secret-token stt-ru serve --host 0.0.0.0 -j 2    # network access with a token
```

The REST API reference with request examples is at `/docs`.

## Web interface

*   Upload by drag and drop or by choosing files, several files in a row, an upload progress bar.
*   The list of jobs of this browser with their state: queued and the place in the queue, running, done, failed.
*   A waveform per channel with the segments of each speaker, a player and a table of segments: clicking a row or the waveform jumps to that place in the recording, and the current segment is highlighted.
*   Downloading the result in TXT, JSON, JSONL, SRT, VTT, ASS and CSV, deleting a job.

Jobs are visible in the browser they were uploaded from: their list is kept in the browser, and job identifiers are random.

## Parameters

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--host` | `127.0.0.1` | Address: `127.0.0.1` - this computer only, `0.0.0.0` - all addresses |
| `--port` | `8000` | Port |
| `-j`, `--workers` | `1` | How many files to transcribe at once |
| `--max-size` | `500` | Largest file size, MB |
| `--max-duration` | `180` | Longest recording, minutes |
| `--max-queue` | `100` | How many files can wait in the queue |
| `--keep` | `24` | How many hours to keep a file and its result after transcription |
| `--data-dir` | the system temporary folder | Folder for uploaded and temporary files |
| `-q`, `--quiet` | off | Do not print progress; errors are always printed |

The model and speech detection parameters are on the [Models](models.md#options) page. The address and parameters are checked before the model is loaded: a busy port gives a message and exit code 1.

## Access token { #token }

The token is set by the `STT_RU_TOKEN` environment variable (a command line parameter would be visible to everyone in the process list). With a token, all job actions require the `Authorization: Bearer <token>` header; the web interface asks for the token once and keeps it in a cookie for 30 days. Without a token, the page, `/healthz`, `/v1/info` and the API reference are open. The token is a string of Latin letters, digits, `-` and `_`: many HTTP clients do not send Cyrillic in headers.

Without a token anyone who can reach the port can use the service, so with `--host 0.0.0.0` the service prints a warning. There is one token for everyone: these are not user accounts. Over plain HTTP the token travels in clear text, so outside a trusted network put the service behind an HTTPS proxy (nginx, Caddy). A random token:

``` bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

## Limits and storage { #limits }

*   A file larger than `--max-size` is rejected during upload (response 413) and is not written to disk.
*   A recording longer than `--max-duration` gives a failed job; the duration is checked from the file header or while decoding, so a long recording is not read into memory entirely.
*   If `--max-queue` files are waiting, new ones get a 503 response with `Retry-After`. An upload without the token (401) or into a full queue is refused before the file is sent.
*   Files are kept in a temporary folder under random names and deleted on request, `--keep` hours after the job finishes and when the service stops. Jobs are kept in memory and are lost on restart; the files of a run that crashed (`kill -9`, out of memory) are deleted on the next start with the same folder (except on Windows).

## REST API { #api }

| Method and path | What it does |
| :-------------- | :----------- |
| `POST /v1/transcriptions` | Queues a file (multipart, field `file`): response 202 with the job and a `Location` header |
| `GET /v1/jobs/{id}` | Job state |
| `GET /v1/jobs/{id}/result?format=srt` | Result in a format (`json` by default) with the file name in `Content-Disposition` |
| `GET /v1/jobs/{id}/waveform` | Waveform peaks per channel |
| `GET /v1/jobs/{id}/audio` | The uploaded file, with seeking support (`Range`): audio and video for the player, anything else as an attachment |
| `DELETE /v1/jobs/{id}` | Deletes the job and its files |
| `GET /v1/info` | Version, model, limits, formats, whether a token is required |
| `GET /healthz` | Health check: `{"status": "ok"}` |

A job is JSON with the fields `id`, `status` (`queued`, `running`, `done`, `failed`), `filename`, `size`, `created`, `started`, `finished`, `expires` (when the job will be deleted), `position` (how many jobs are ahead), `duration`, `channels`, `segments` (number of segments) and `error`. Errors are `{"detail": "..."}` with messages in Russian: 401 without a token, 404 for a deleted job, 409 for a job that is not ready or failed, 413 for a large file, 503 when the queue is full.

``` bash
token="Authorization: Bearer my-secret-token"
job=$(curl -s -H "$token" -F file=@call.wav http://127.0.0.1:8000/v1/transcriptions | jq -r .id)
curl -s -H "$token" http://127.0.0.1:8000/v1/jobs/$job | jq .status
curl -s -H "$token" -OJ "http://127.0.0.1:8000/v1/jobs/$job/result?format=srt"
```

``` python
import time

import requests

api = "http://127.0.0.1:8000/v1"
headers = {"Authorization": "Bearer my-secret-token"}
with open("call.wav", "rb") as file:
    job = requests.post(f"{api}/transcriptions", files={"file": file}, headers=headers).json()
while job["status"] in ("queued", "running"):
    time.sleep(1)
    job = requests.get(f"{api}/jobs/{job['id']}", headers=headers).json()
result = requests.get(f"{api}/jobs/{job['id']}/result", headers=headers).json()
for segment in result["segments"]:
    print(segment["speaker"], segment["text"])
```

## The service in your application { #embedding }

`create_app` returns an ASGI (FastAPI) application with a `JobQueue`. Run it with your own ASGI server at the root of an address: the page requests `/static` and `/v1` from the root, and cleaning up expired jobs and closing the queue happen on the startup and shutdown of the application itself, so it cannot be mounted under a path inside another application.

``` python
import uvicorn

from stt_ru import Transcriber
from stt_ru.service import JobQueue, create_app

jobs = JobQueue(Transcriber(), "/data/stt", workers=2, max_duration=3600, keep=3600)
app = create_app(jobs, token="my-secret-token", max_size=200 << 20)
uvicorn.run(app, host="0.0.0.0", port=8000)
```
