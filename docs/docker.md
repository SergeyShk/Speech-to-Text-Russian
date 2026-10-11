# Docker

## Образы

Образы публикуются в GitHub Container Registry при каждом выпуске:

| Образ | Платформы | Для чего |
| :---- | :-------- | :------- |
| `ghcr.io/sergeyshk/stt-ru:<версия>`, `:latest` | linux/amd64, linux/arm64 | CPU |
| `ghcr.io/sergeyshk/stt-ru:<версия>-gpu`, `:latest-gpu` | linux/amd64 | NVIDIA GPU |

В образе - stt-ru со всеми дополнениями (сервис, оценка качества, ffmpeg) и модель по умолчанию: контейнер распознает без сети. Команда по умолчанию - `stt-ru serve --host 0.0.0.0`, вместо нее можно передать любую команду `stt-ru`. Рабочий каталог - `/data`, пользователь - `stt` с uid 1000.

## Веб-сервис

``` bash
docker run -d --name stt-ru -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru
```

Веб-интерфейс - на `http://<адрес сервера>:8000`. Параметры [сервиса](service.md) передаются после имени образа вместе с командой:

``` bash
docker run -d -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru serve --host 0.0.0.0 -j 2 --max-size 1000
```

`docker stop` останавливает сервис штатно: образ останавливается сигналом SIGINT, как Ctrl+C. Состояние контейнера показывает healthcheck: у `serve` - ответ `/healthz`, у других команд контейнер здоров, пока работает.

## Команды для файлов { #commands }

Каталог с записями монтируется в `/data`:

``` bash
docker run --rm -v "$PWD:/data" ghcr.io/sergeyshk/stt-ru transcribe call.wav
docker run --rm -v "$PWD:/data" --user "$(id -u):$(id -g)" ghcr.io/sergeyshk/stt-ru transcribe calls/ -o out/ -f srt,csv
docker run -d -v /archive:/data --user "$(id -u):$(id -g)" ghcr.io/sergeyshk/stt-ru transcribe wav -r -o out --watch --delete-source
```

Команды в контейнере работают от пользователя с uid 1000. Чтобы записывать результаты (`-o`, `--summary`) или удалять исходники в каталоге другого пользователя, добавьте `--user "$(id -u):$(id -g)"`.

## Другие модели { #models }

Другие модели скачиваются при первом запуске в `/opt/huggingface` внутри контейнера и пропадают вместе с ним. Чтобы не скачивать их заново после пересоздания контейнера, подключите к `/opt/huggingface` именованный том: при первом подключении Docker скопирует в него модель из образа. Том хранит файлы и после обновления образа; если в новой версии сменится модель по умолчанию, она скачается в том.

``` bash
docker run -d -p 8000:8000 -e STT_RU_TOKEN=my-secret-token -v stt-models:/opt/huggingface ghcr.io/sergeyshk/stt-ru serve --host 0.0.0.0 -m t-tech/t-one
```

## Docker Compose

`docker/compose.yaml` в репозитории запускает сервис с портом 8000, токеном и перезапуском. Токен берется из переменной окружения или файла `.env` рядом с файлом Compose:

``` bash
echo "STT_RU_TOKEN=my-secret-token" > docker/.env
docker compose -f docker/compose.yaml up -d
```

## GPU { #gpu }

Образ `-gpu` распознает на NVIDIA GPU через CUDA 13. Нужны:

*   GPU поколения Turing (compute capability 7.5) или новее: CUDA 13 не поддерживает Maxwell, Pascal и Volta;
*   драйвер NVIDIA 580 или новее;
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

``` bash
docker run -d --gpus all -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru:latest-gpu
docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml up -d
```

Без GPU образ распознает на CPU.

## Сборка образа

``` bash
make docker        # docker build -f docker/Dockerfile -t stt-ru .
make docker-gpu    # docker build -f docker/Dockerfile.gpu -t stt-ru:gpu .
```

Аргумент сборки `MODEL` меняет модель, которая загружается в образ; пустое значение собирает образ без модели. Команды образа по-прежнему берут модель по умолчанию, поэтому загруженную модель указывают параметром `-m`:

``` bash
docker build -f docker/Dockerfile --build-arg MODEL=t-tech/t-one -t stt-ru:t-one .
docker run -d -p 8000:8000 -e STT_RU_TOKEN=my-secret-token stt-ru:t-one serve --host 0.0.0.0 -m t-tech/t-one
```
