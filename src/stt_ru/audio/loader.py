import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from ..exceptions import AudioError

Samples = npt.NDArray[np.float32]


@dataclass(frozen=True, eq=False)
class Audio:
    """
    Аудиозапись в памяти

    Описание:
        Отсчеты хранятся в float32 в диапазоне от -1 до 1, по столбцу на канал

    Аргументы:
        samples (numpy.ndarray): Отсчеты формы (число отсчетов, число каналов)
        sample_rate (int): Частота дискретизации, Гц

    Исключения:
        AudioError: Если отсчеты не двумерный массив float32, в нем нет каналов
            или частота дискретизации не положительна

    Пример использования:
        >>> import numpy as np
        >>> from stt_ru.audio import Audio
        >>> audio = Audio(np.zeros((16000, 2), dtype=np.float32), 16000)
        >>> audio.num_channels, audio.duration
        (2, 1.0)
    """

    samples: Samples
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 2 or self.samples.dtype != np.float32:
            raise AudioError(
                "Отсчеты должны быть массивом float32 формы (число отсчетов, число каналов)"
            )
        if self.samples.shape[1] == 0:
            raise AudioError("В записи нет каналов")
        if self.sample_rate <= 0:
            raise AudioError(
                f"Частота дискретизации должна быть положительной: {self.sample_rate}"
            )

    @property
    def num_channels(self) -> int:
        """Число каналов"""
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        """Длительность, с"""
        return int(self.samples.shape[0]) / self.sample_rate

    def channel(self, index: int) -> Samples:
        """
        Отсчеты одного канала

        Аргументы:
            index (int): Номер канала, с нуля

        Вывод:
            numpy.ndarray: Одномерный массив отсчетов канала

        Исключения:
            AudioError: Если канала с таким номером нет

        Пример использования:
            >>> import numpy as np
            >>> from stt_ru.audio import Audio
            >>> audio = Audio(np.array([[0.1, 0.5], [0.2, 0.6]], dtype=np.float32), 8000)
            >>> audio.channel(1)
            array([0.5, 0.6], dtype=float32)
        """
        if not 0 <= index < self.num_channels:
            raise AudioError(f"Канала {index} нет: в записи каналов {self.num_channels}")
        return np.ascontiguousarray(self.samples[:, index])

    def mono(self) -> Samples:
        """
        Каналы, сведенные в один

        Вывод:
            numpy.ndarray: Одномерный массив средних по каналам

        Пример использования:
            >>> import numpy as np
            >>> from stt_ru.audio import Audio
            >>> audio = Audio(np.array([[0.1, 0.5], [0.2, 0.6]], dtype=np.float32), 8000)
            >>> audio.mono()
            array([0.3, 0.4], dtype=float32)
        """
        mixed: Samples = self.samples.mean(axis=1, dtype=np.float32)
        return mixed


def find_ffmpeg(ffmpeg: str | os.PathLike[str] | None = None) -> str | None:
    """
    Поиск программы ffmpeg

    Описание:
        Порядок поиска: аргумент, программа ffmpeg из PATH, сборка из пакета imageio-ffmpeg
        (устанавливается с stt-ru[ffmpeg])

    Аргументы:
        ffmpeg (str): Путь к программе ffmpeg

    Вывод:
        str: Путь к ffmpeg; None, если программа не найдена
    """
    if ffmpeg is not None:
        return os.fspath(ffmpeg)
    if found := shutil.which("ffmpeg"):
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except RuntimeError:
        return None


def load_audio(
    path: str | os.PathLike[str],
    *,
    sample_rate: int | None = None,
    ffmpeg: str | os.PathLike[str] | None = None,
) -> Audio:
    """
    Чтение аудиофайла

    Описание:
        WAV, FLAC, OGG и MP3 читает библиотека soundfile. Остальные форматы, в том числе
        звуковая дорожка видео, и смена частоты дискретизации требуют ffmpeg (см. find_ffmpeg)

    Аргументы:
        path (str): Путь к файлу
        sample_rate (int): Частота дискретизации результата, Гц; None - частота файла
        ffmpeg (str): Путь к программе ffmpeg; None - поиск через find_ffmpeg

    Вывод:
        Audio: Запись со всеми каналами файла

    Исключения:
        FileNotFoundError: Если файла нет
        AudioError: Если файл не читается или для него нужен ffmpeg, а ffmpeg не найден

    Пример использования:
        >>> from stt_ru.audio import load_audio
        >>> audio = load_audio(call_wav)
        >>> audio.num_channels, audio.sample_rate, audio.duration
        (2, 8000, 2.0)
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Файл не найден: {source}")
    try:
        samples, rate = sf.read(source, dtype="float32", always_2d=True)
    except sf.LibsndfileError as error:
        reason = f"формат не читается библиотекой soundfile ({error})"
        samples, rate = _decode(source, sample_rate, _require_ffmpeg(ffmpeg, reason))
    else:
        if sample_rate is not None and rate != sample_rate:
            reason = f"нужна смена частоты дискретизации с {rate} на {sample_rate} Гц"
            samples, rate = _decode(source, sample_rate, _require_ffmpeg(ffmpeg, reason))
    return Audio(samples, rate)


def _require_ffmpeg(ffmpeg: str | os.PathLike[str] | None, reason: str) -> str:
    found = find_ffmpeg(ffmpeg)
    if found is None:
        raise AudioError(
            f"Для файла {reason}, а ffmpeg не найден: установите ffmpeg "
            "или пакет с ним: pip install 'stt-ru[ffmpeg]'"
        )
    return found


def _decode(path: Path, sample_rate: int | None, ffmpeg: str) -> tuple[Samples, int]:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "audio.wav"
        # Префикс file: не дает ffmpeg принять имя файла за протокол вроде concat: или http:
        command = [ffmpeg, "-nostdin", "-v", "error", "-i", f"file:{path.resolve()}", "-vn"]
        command += ["-c:a", "pcm_f32le"]
        if sample_rate is not None:
            command += ["-ar", str(sample_rate)]
        command.append(str(output))
        try:
            result = subprocess.run(
                command, capture_output=True, encoding="utf-8", errors="replace", check=False
            )
        except OSError as error:
            raise AudioError(f"Не удалось запустить ffmpeg ({ffmpeg}): {error}") from error
        if result.returncode != 0:
            raise AudioError(f"ffmpeg не прочитал файл {path}: {result.stderr.strip()}")
        samples: Samples
        rate: int
        samples, rate = sf.read(output, dtype="float32", always_2d=True)
    return samples, rate
