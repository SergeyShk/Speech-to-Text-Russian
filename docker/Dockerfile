# syntax=docker/dockerfile:1
# Образ stt-ru для CPU: docker build -f docker/Dockerfile -t stt-ru .

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project \
        --extra service --extra ffmpeg --extra eval
COPY README.md LICENSE.txt ./
COPY src src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable --extra service --extra ffmpeg --extra eval

FROM python:3.12-slim
ARG MODEL=gigaam-v3-e2e-rnnt
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface
RUN useradd --uid 1000 --create-home stt \
    && mkdir -p /opt/huggingface /data \
    && chown stt:stt /opt/huggingface /data
COPY --from=build /opt/venv /opt/venv
USER stt
# Модель по умолчанию загружается при сборке: контейнер распознает без сети
RUN if [ -n "$MODEL" ]; then \
        python -c "import sys; from stt_ru import Transcriber; Transcriber(sys.argv[1], providers=['CPUExecutionProvider'])" "$MODEL"; \
    fi
WORKDIR /data
EXPOSE 8000
# SIGINT, а не SIGTERM: так останавливаются без потерь и сервис, и пакетное распознавание
STOPSIGNAL SIGINT
COPY docker/healthcheck.py /opt/stt-ru/healthcheck.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --start-interval=2s \
    CMD ["python", "/opt/stt-ru/healthcheck.py"]
ENTRYPOINT ["stt-ru"]
CMD ["serve", "--host", "0.0.0.0"]
