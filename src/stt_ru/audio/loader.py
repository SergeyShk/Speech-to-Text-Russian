import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

from ..exceptions import AudioError, ParameterError

Samples = npt.NDArray[np.float32]

# SF_COUNT_MAX: libsndfile не знает длину записи
_UNKNOWN_FRAMES = 2**63 - 1
# Больше каналов бывает у массива формы (число каналов, число отсчетов), а не у записи
MAX_CHANNELS = 64
_FFMPEG_PREFIX = re.compile(r"^\[[^\]]*\]\s*")


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
        AudioError: Если отсчеты не двумерный массив float32, в нем нет каналов или их
            больше MAX_CHANNELS, или частота дискретизации не положительна

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
        if self.samples.shape[1] > MAX_CHANNELS:
            raise AudioError(
                f"Каналов {self.samples.shape[1]}, больше {MAX_CHANNELS}: отсчеты должны иметь "
                "форму (число отсчетов, число каналов); массив (каналы, отсчеты) транспонируйте"
            )
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

    def peaks(self, points: int) -> Samples:
        """
        Пики для рисования формы волны

        Описание:
            Запись делится на points равных частей, у каждой части и канала берется
            наибольшая амплитуда по модулю. Если отсчетов меньше points, частей столько же,
            сколько отсчетов

        Аргументы:
            points (int): Число частей

        Вывод:
            numpy.ndarray: Массив формы (число каналов, число частей)

        Исключения:
            ParameterError: Если points меньше 1

        Пример использования:
            >>> import numpy as np
            >>> from stt_ru.audio import Audio
            >>> samples = np.array([[0.1, -0.5], [-0.4, 0.2], [0.3, 0.0]], dtype=np.float32)
            >>> Audio(samples, 8000).peaks(2)
            array([[0.4, 0.3],
                   [0.5, 0. ]], dtype=float32)
        """
        if points < 1:
            raise ParameterError(f"Число частей должно быть не меньше 1: {points}")
        frames = len(self.samples)
        size = max(1, -(-frames // points))
        peaks = np.zeros((self.num_channels, -(-frames // size)), dtype=np.float32)
        # Блоками, чтобы модуль отсчетов не копировал всю запись
        block = max(1, (1 << 20) // size) * size
        for start in range(0, frames, block):
            chunk = np.abs(self.samples[start : start + block])
            found = np.maximum.reduceat(chunk, np.arange(0, len(chunk), size), axis=0)
            peaks[:, start // size : start // size + len(found)] = found.T
        return peaks


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
    max_duration: float | None = None,
) -> Audio:
    """
    Чтение аудиофайла

    Описание:
        WAV, FLAC, OGG и MP3 читает библиотека soundfile. Остальные форматы, в том числе
        звуковая дорожка видео, смена частоты дискретизации и файлы без длины записи
        в заголовке требуют ffmpeg (см. find_ffmpeg). MP3 и файлы с нулевой длиной
        в заголовке декодирует ffmpeg, если он найден: soundfile читает их только
        до длины из заголовка

    Аргументы:
        path (str): Путь к файлу
        sample_rate (int): Частота дискретизации результата, Гц; None - частота файла
        ffmpeg (str): Путь к программе ffmpeg; None - поиск через find_ffmpeg
        max_duration (float): Наибольшая длительность записи, с, конечное положительное
            число; None - без ограничения. Длинная запись не декодируется целиком:
            длительность проверяется по заголовку, а ffmpeg останавливается после max_duration

    Вывод:
        Audio: Запись со всеми каналами файла

    Исключения:
        FileNotFoundError: Если файла нет
        AudioError: Если частота дискретизации не положительна, файл пуст или не читается,
            в нем нет звука, для него нужен ffmpeg, а ffmpeg не найден, или запись длиннее
            max_duration
        ParameterError: Если max_duration не конечное положительное число

    Пример использования:
        >>> from stt_ru.audio import load_audio
        >>> audio = load_audio(call_wav)
        >>> audio.num_channels, audio.sample_rate, audio.duration
        (2, 8000, 2.0)
    """
    if sample_rate is not None and sample_rate <= 0:
        raise AudioError(f"Частота дискретизации должна быть положительной: {sample_rate}")
    if max_duration is not None and not (math.isfinite(max_duration) and max_duration > 0):
        raise ParameterError(
            f"Наибольшая длительность должна быть конечным положительным числом: {max_duration}"
        )
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Файл не найден: {source}")
    if source.stat().st_size == 0:
        raise AudioError(f"Файл пуст: {source}")
    audio = _read(source, sample_rate, ffmpeg, max_duration)
    if max_duration is not None:
        _check_duration(audio.duration, max_duration)
    return audio


def _read(
    source: Path,
    sample_rate: int | None,
    ffmpeg: str | os.PathLike[str] | None,
    max_duration: float | None,
) -> Audio:
    try:
        with sf.SoundFile(source) as file:
            # Длина MP3 в заголовке - оценка: ее ограничивает ffmpeg (-t) или проверка после чтения
            known = 0 < file.frames < _UNKNOWN_FRAMES and file.format != "MP3"
            if max_duration is not None and known:
                _check_duration(file.frames / file.samplerate, max_duration)
            reason = _ffmpeg_reason(file, sample_rate, ffmpeg)
            if reason is None:
                samples: Samples = file.read(dtype="float32", always_2d=True)
                return Audio(samples, file.samplerate)
    except sf.LibsndfileError as error:
        reason = f"формат не читается библиотекой soundfile ({error})"
    decoded, rate = _decode(source, sample_rate, _require_ffmpeg(ffmpeg, reason), max_duration)
    return Audio(decoded, rate)


def _check_duration(duration: float, limit: float) -> None:
    if duration > limit:
        text = f"{limit / 60:g} мин" if limit >= 60 else f"{limit:g} с"
        raise AudioError(f"Запись длиннее {text}")


def _ffmpeg_reason(
    file: sf.SoundFile, sample_rate: int | None, ffmpeg: str | os.PathLike[str] | None
) -> str | None:
    if file.frames >= _UNKNOWN_FRAMES:
        return "длина записи не указана в заголовке"
    if sample_rate is not None and file.samplerate != sample_rate:
        return f"нужна смена частоты дискретизации с {file.samplerate} на {sample_rate} Гц"
    if (file.format == "MP3" or file.frames == 0) and find_ffmpeg(ffmpeg) is not None:
        return "длина записи в заголовке ненадежна"
    return None


def _require_ffmpeg(ffmpeg: str | os.PathLike[str] | None, reason: str) -> str:
    found = find_ffmpeg(ffmpeg)
    if found is None:
        raise AudioError(
            f"Для файла {reason}, а ffmpeg не найден: установите ffmpeg "
            "или пакет с ним: pip install 'stt-ru[ffmpeg]'"
        )
    return found


def _decode(
    path: Path, sample_rate: int | None, ffmpeg: str, max_duration: float | None = None
) -> tuple[Samples, int]:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "audio.wav"
        # Префикс file: не дает ffmpeg принять имя файла за протокол вроде concat: или http:
        command = [ffmpeg, "-nostdin", "-v", "error", "-i", f"file:{path.resolve()}"]
        command += ["-map", "0:a:0"]
        # RF64 нужен записям длиннее 4 ГиБ: обычный WAV ffmpeg молча обрезает
        command += ["-c:a", "pcm_f32le", "-rf64", "auto"]
        if sample_rate is not None:
            command += ["-ar", str(sample_rate)]
        if max_duration is not None:
            # Секунда сверх предела отличает слишком длинную запись от записи ровно на предел
            command += ["-t", f"{max_duration + 1:.3f}"]
        command.append(str(output))
        try:
            result = subprocess.run(
                command, capture_output=True, encoding="utf-8", errors="replace", check=False
            )
        except OSError as error:
            raise AudioError(f"Не удалось запустить ffmpeg ({ffmpeg}): {error}") from error
        if result.returncode != 0:
            message = _ffmpeg_error(result.stderr).replace(f"file:{path.resolve()}", str(path))
            raise AudioError(f"ffmpeg не прочитал файл {path}: {message}")
        samples: Samples
        rate: int
        samples, rate = sf.read(output, dtype="float32", always_2d=True)
    return samples, rate


def _ffmpeg_error(stderr: str) -> str:
    if "matches no streams" in stderr:
        return "в файле нет звуковой дорожки"
    # Первая строка без адреса вида [in#0 @ 0x...]: дальше ffmpeg повторяет ее с путями
    for line in stderr.splitlines():
        if text := _FFMPEG_PREFIX.sub("", line).strip():
            return text
    return "неизвестная ошибка"
