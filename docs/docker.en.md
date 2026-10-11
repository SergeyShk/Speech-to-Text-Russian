# Docker

## Images

Images are published to the GitHub Container Registry with every release:

| Image | Platforms | Used for |
| :---- | :-------- | :------- |
| `ghcr.io/sergeyshk/stt-ru:<version>`, `:latest` | linux/amd64, linux/arm64 | CPU |
| `ghcr.io/sergeyshk/stt-ru:<version>-gpu`, `:latest-gpu` | linux/amd64 | NVIDIA GPU |

The image contains stt-ru with all extras (the service, quality evaluation, ffmpeg) and the default model: the container recognizes speech without network. The default command is `stt-ru serve --host 0.0.0.0`; any `stt-ru` command can be passed instead. The working folder is `/data`, and the user is `stt` with uid 1000.

## Web service

``` bash
docker run -d --name stt-ru -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru
```

The web interface is at `http://<server address>:8000`. The [service](service.md) parameters are passed after the image name together with the command:

``` bash
docker run -d -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru serve --host 0.0.0.0 -j 2 --max-size 1000
```

`docker stop` stops the service cleanly: the image stops with the SIGINT signal, like Ctrl+C. The container state comes from a healthcheck: for `serve` it is the `/healthz` response, and with other commands the container is healthy while it runs.

## Commands for files { #commands }

The folder with recordings is mounted to `/data`:

``` bash
docker run --rm -v "$PWD:/data" ghcr.io/sergeyshk/stt-ru transcribe call.wav
docker run --rm -v "$PWD:/data" --user "$(id -u):$(id -g)" ghcr.io/sergeyshk/stt-ru transcribe calls/ -o out/ -f srt,csv
docker run -d -v /archive:/data --user "$(id -u):$(id -g)" ghcr.io/sergeyshk/stt-ru transcribe wav -r -o out --watch --delete-source
```

Commands in the container run as the user with uid 1000. To write results (`-o`, `--summary`) or delete sources in a folder of another user, add `--user "$(id -u):$(id -g)"`.

## Other models { #models }

Other models are downloaded on the first run to `/opt/huggingface` inside the container and disappear with it. To avoid downloading them again after the container is recreated, mount a named volume to `/opt/huggingface`: on the first mount Docker copies the model from the image into it. The volume keeps its files after the image is updated; if the default model changes in a new version, it is downloaded into the volume.

``` bash
docker run -d -p 8000:8000 -e STT_RU_TOKEN=my-secret-token -v stt-models:/opt/huggingface ghcr.io/sergeyshk/stt-ru serve --host 0.0.0.0 -m t-tech/t-one
```

## Docker Compose

`docker/compose.yaml` in the repository runs the service with port 8000, a token and restarts. The token comes from an environment variable or an `.env` file next to the Compose file:

``` bash
echo "STT_RU_TOKEN=my-secret-token" > docker/.env
docker compose -f docker/compose.yaml up -d
```

## GPU { #gpu }

The `-gpu` image recognizes speech on NVIDIA GPUs with CUDA 13. It needs:

*   a Turing (compute capability 7.5) or newer GPU: CUDA 13 does not support Maxwell, Pascal and Volta;
*   NVIDIA driver 580 or newer;
*   the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

``` bash
docker run -d --gpus all -p 8000:8000 -e STT_RU_TOKEN=my-secret-token ghcr.io/sergeyshk/stt-ru:latest-gpu
docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml up -d
```

Without a GPU the image recognizes speech on the CPU.

## Building an image

``` bash
make docker        # docker build -f docker/Dockerfile -t stt-ru .
make docker-gpu    # docker build -f docker/Dockerfile.gpu -t stt-ru:gpu .
```

The `MODEL` build argument changes the model downloaded into the image; an empty value builds an image without a model. The image commands still use the default model, so the downloaded model is chosen with `-m`:

``` bash
docker build -f docker/Dockerfile --build-arg MODEL=t-tech/t-one -t stt-ru:t-one .
docker run -d -p 8000:8000 -e STT_RU_TOKEN=my-secret-token stt-ru:t-one serve --host 0.0.0.0 -m t-tech/t-one
```
