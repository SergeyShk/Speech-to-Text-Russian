# Установка

Библиотеку и команду `stt-ru` можно установить с помощью `pip` или из репозитория проекта через `git`. Готовый Docker-образ описан на странице [Docker](docker.md).

## Зависимости

*   `python` - 3.11 и выше
*   [`onnx-asr`](https://github.com/istupakov/onnx-asr) - 0.12, с ONNX Runtime для CPU и загрузкой моделей с Hugging Face
*   `numpy`
*   `soundfile`

## Установка с помощью `pip`

``` bash
pip install stt-ru
```

Дополнительные возможности ставятся дополнениями:

| Дополнение | Что добавляет | Пакеты |
| :--------- | :------------ | :----- |
| `ffmpeg` | Форматы M4A, AAC, AMR, WMA, звук из видео и смену частоты без установленного ffmpeg | `imageio-ffmpeg` |
| `service` | [Веб-интерфейс и REST API](service.md): команда `stt-ru serve` | `fastapi`, `uvicorn`, `python-multipart` |
| `eval` | [Оценку качества](evaluation.md): команда `stt-ru evaluate` | `jiwer` |

``` bash
pip install "stt-ru[ffmpeg,service,eval]"
```

Или с помощью [uv](https://docs.astral.sh/uv/):

``` bash
uv add "stt-ru[ffmpeg]"
```

!!! tip "ffmpeg"
    WAV, FLAC, OGG (в том числе Opus), MP3, AIFF, AU, CAF и W64 читает библиотека soundfile. M4A, AAC, AMR, WMA, звуку из видео, записям с частотой, которую модель не принимает, и файлам без длины записи в заголовке нужен ffmpeg. stt-ru ищет его в `PATH`, затем в пакете `imageio-ffmpeg` из дополнения `ffmpeg`; путь можно задать параметром `--ffmpeg`. Если ffmpeg нужен, но не найден, файл не распознается, а в сообщении будет подсказка, что установить.

## Модели { #models }

Модель скачивается с Hugging Face при первом распознавании и хранится в его кэше: модель по умолчанию `gigaam-v3-e2e-rnnt` в квантизации int8 вместе с Silero VAD занимает около 230 МБ. Кэш лежит в `~/.cache/huggingface`, другой каталог задает переменная окружения `HF_HOME`:

``` bash
export HF_HOME=/data/models
```

Модель из кэша работает без сети. Для компьютера без интернета скопируйте на него весь каталог кэша - в нем и модель, и Silero VAD - и укажите его в `HF_HOME` (см. также [Модели](models.md#local)).

## GPU и другие устройства { #devices }

По умолчанию модель работает на CPU через пакет `onnxruntime`. Для других устройств его заменяют другой сборкой ONNX Runtime: для NVIDIA GPU - той же версии, для DirectML - последней (onnx-asr работает с ONNX Runtime 1.18.1 и новее):

| Устройство | `--device` | Пакет | Условия |
| :--------- | :--------- | :---- | :------ |
| NVIDIA GPU | `cuda` | `onnxruntime-gpu` | CUDA 13 и cuDNN 9 для ONNX Runtime 1.31 |
| DirectML (Windows) | `directml` | `onnxruntime-directml` | Windows 10 и новее |
| Apple CoreML | `coreml` | `onnxruntime` для macOS | только явно: на этих моделях CoreML медленнее CPU |

Замена для NVIDIA GPU:

``` bash
pip show onnxruntime | grep Version    # версия, например 1.31.0
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.31.0
```

Требования сборок ONNX Runtime к драйверам и библиотекам - в [документации ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/). По умолчанию (`--device auto`) stt-ru берет доступные устройства в порядке приоритета ONNX Runtime - GPU раньше CPU, - кроме CoreML и TensorRT; недоступное устройство, указанное явно, - ошибка с подсказкой, какой пакет поставить. Для NVIDIA GPU есть готовый [Docker-образ](docker.md#gpu).

## Установка с помощью `git`

Последняя версия из репозитория проекта:

``` bash
git clone https://github.com/SergeyShk/Speech-to-Text-Russian.git
cd Speech-to-Text-Russian
pip install ".[ffmpeg,service,eval]"
```

Для разработки удобнее [uv](https://docs.astral.sh/uv/): он создаст окружение со всеми зависимостями и инструментами разработки.

``` bash
uv sync --all-groups --no-group notebooks --all-extras
```

Порядок работы и проверки перед пул-реквестом описаны в [CONTRIBUTING.md](https://github.com/SergeyShk/Speech-to-Text-Russian/blob/master/CONTRIBUTING.md).
