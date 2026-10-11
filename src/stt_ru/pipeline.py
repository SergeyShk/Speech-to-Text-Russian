import os
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf

from .asr import DEFAULT_MODEL, DEFAULT_QUANTIZATION, SAMPLE_RATES, Recognizer
from .audio import Audio, load_audio
from .audio.loader import Samples
from .result import Segment, Transcript
from .vad import VadOptions

RESAMPLE_RATE = 16000
# Окна ограничивают память: onnx-asr передискретизирует запись целиком
WINDOW_SECONDS = 600.0
CUT_SEARCH_SECONDS = 5.0
CUT_FRAME_SECONDS = 0.02
CUT_QUIET_SECONDS = 0.3
CUT_FLOOR_DB = 40.0
# Остаток ниже -30 дБ после подбора громкости и сдвига: одна запись в нескольких каналах
SAME_CHANNEL_RATIO = 1e-3
SAME_CHANNEL_LAG = 0.01
LAG_FRAME = 1 << 14
LAG_FRAMES = 16

Track = tuple[int | None, Samples]


class Transcriber:
    """
    Распознавание аудиофайлов

    Описание:
        Каналы стереозаписи распознаются отдельно: в записях разговоров каждый канал - свой
        говорящий; каналы, которые отличаются только громкостью и сдвигом до 10 мс
        (см. same_signal), распознаются один раз. Файл с частотой, которую модель
        не принимает, и формат, который читает только ffmpeg, переводятся в 16 кГц через
        ffmpeg (см. load_audio). Записи длиннее 10 минут распознаются окнами около 10 минут
        с границей в паузе (см. cut_points)

    Аргументы:
        model (str): Имя модели onnx-asr (см. Recognizer)
        quantization (str): Квантизация весов: "int8" или None - исходные веса
        model_dir (str): Каталог с файлами модели
        providers (list[str]): Провайдеры ONNX Runtime; None - default_providers()
        vad (VadOptions): Параметры поиска речи
        split_channels (bool): Распознавать каналы отдельно; False - сводить в один
        ffmpeg (str): Путь к ffmpeg; None - поиск через find_ffmpeg

    Исключения:
        ModelError: Если модель не загружается

    Пример использования:
        >>> from stt_ru import Transcriber
        >>> transcript = Transcriber().transcribe("call.wav")  # doctest: +SKIP
        >>> for segment in transcript.segments:  # doctest: +SKIP
        ...     print(segment.speaker, segment.start, segment.end, segment.text)
        Канал 1 0.994 9.054 Наталья, благодарю вас за ожидание. ...
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        quantization: str | None = DEFAULT_QUANTIZATION,
        model_dir: str | os.PathLike[str] | None = None,
        providers: Sequence[str] | None = None,
        vad: VadOptions | None = None,
        split_channels: bool = True,
        ffmpeg: str | os.PathLike[str] | None = None,
    ) -> None:
        self.recognizer = Recognizer(
            model, quantization=quantization, model_dir=model_dir, providers=providers, vad=vad
        )
        self.split_channels = split_channels
        self.ffmpeg = ffmpeg

    def transcribe(self, path: str | os.PathLike[str]) -> Transcript:
        """
        Распознавание аудиофайла

        Аргументы:
            path (str): Путь к файлу (см. load_audio)

        Вывод:
            Transcript: Фрагменты речи по времени начала

        Исключения:
            FileNotFoundError: Если файла нет
            AudioError: Если файл не читается
        """
        return self.transcribe_audio(self.load(path), source=Path(path).name)

    def load(self, path: str | os.PathLike[str], *, max_duration: float | None = None) -> Audio:
        """
        Чтение аудиофайла с частотой, которую принимает модель

        Описание:
            Файл с другой частотой и формат, который читает только ffmpeg, декодируются
            сразу в 16 кГц (см. load_audio)

        Аргументы:
            path (str): Путь к файлу
            max_duration (float): Наибольшая длительность записи, с; None - без ограничения

        Вывод:
            Audio: Запись для transcribe_audio

        Исключения:
            FileNotFoundError: Если файла нет
            AudioError: Если файл не читается или запись длиннее max_duration
        """
        # Именованный канал без писателя заблокировал бы sf.info навсегда
        if not Path(path).is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")
        try:
            native: int | None = sf.info(os.fspath(path)).samplerate
        except sf.LibsndfileError:
            native = None
        rate = None if native in SAMPLE_RATES else RESAMPLE_RATE
        return load_audio(path, sample_rate=rate, ffmpeg=self.ffmpeg, max_duration=max_duration)

    def transcribe_audio(self, audio: Audio, source: str | None = None) -> Transcript:
        """
        Распознавание записи в памяти

        Аргументы:
            audio (Audio): Запись с частотой из SAMPLE_RATES
            source (str): Имя файла записи для результата

        Вывод:
            Transcript: Фрагменты речи по времени начала; фрагменты без текста отбрасываются

        Исключения:
            AudioError: Если модель не принимает частоту записи
        """
        tracks, channel_map = self._tracks(audio)
        cuts = cut_points([samples for _, samples in tracks], audio.sample_rate)
        segments: list[Segment] = []
        for start, end in pairwise(cuts):
            if end == start:
                continue
            offset = start / audio.sample_rate
            found = self.recognizer.recognize(
                [samples[start:end] for _, samples in tracks], audio.sample_rate
            )
            segments += [
                Segment(round(offset + begin, 3), round(offset + finish, 3), text, channel)
                for (channel, _), window in zip(tracks, found, strict=True)
                for begin, finish, text in window
                if text
            ]
        segments.sort(key=lambda segment: (segment.start, segment.channel or 0))
        return Transcript(
            tuple(segments),
            audio.duration,
            len(tracks),
            self.recognizer.model,
            source,
            channel_map,
        )

    def _tracks(self, audio: Audio) -> tuple[list[Track], tuple[int | None, ...]]:
        single: tuple[int | None, ...] = (None,) * audio.num_channels
        if not self.split_channels:
            return [(None, audio.mono())], single
        lag = int(SAME_CHANNEL_LAG * audio.sample_rate)
        tracks: list[Track] = []
        channel_map: list[int | None] = []
        for index in range(audio.num_channels):
            samples = audio.samples[:, index]
            same = (channel for channel, kept in tracks if same_signal(kept, samples, lag))
            if (match := next(same, None)) is None:
                tracks.append((index, samples))
                channel_map.append(index)
            else:
                channel_map.append(match)
        if len(tracks) == 1:
            return [(None, tracks[0][1])], single
        return tracks, tuple(channel_map)


def same_signal(first: Samples, second: Samples, max_lag: int = 0, block: int = 1 << 20) -> bool:
    """
    Совпадение двух каналов

    Описание:
        Каналы совпадают, если второй - это первый с другой громкостью и сдвигом не больше
        max_lag отсчетов: доля энергии второго канала, которую не объясняет первый,
        не больше SAME_CHANNEL_RATIO (-30 дБ). Сдвиг ищется по LAG_FRAMES самым громким
        частям первого канала. Каналы в противофазе не совпадают, два беззвучных совпадают

    Аргументы:
        first (numpy.ndarray): Отсчеты первого канала
        second (numpy.ndarray): Отсчеты второго канала той же длины
        max_lag (int): Наибольший сдвиг, отсчеты
        block (int): Число отсчетов, которые сравниваются за раз

    Вывод:
        bool: Каналы совпадают

    Пример использования:
        >>> import numpy as np
        >>> from stt_ru.pipeline import same_signal
        >>> signal = np.sin(np.arange(8000, dtype=np.float32))
        >>> same_signal(signal, 0.5 * signal), same_signal(signal, -signal)
        (True, False)
        >>> same_signal(signal[3:], signal[:-3]), same_signal(signal[3:], signal[:-3], max_lag=5)
        (False, True)
    """
    lag = _lag(first, second, max_lag) if max_lag > 0 else 0
    if lag >= 0:
        first, second = first[: len(first) - lag], second[lag:]
    else:
        first, second = first[-lag:], second[: len(second) + lag]
    cross = energy_first = energy_second = 0.0
    for start in range(0, len(first), block):
        a = first[start : start + block].astype(np.float64)
        b = second[start : start + block].astype(np.float64)
        cross += float(np.dot(a, b))
        energy_first += float(np.dot(a, a))
        energy_second += float(np.dot(b, b))
    if energy_first == 0 or energy_second == 0:
        return energy_first == energy_second
    return cross > 0 and 1 - cross**2 / (energy_first * energy_second) <= SAME_CHANNEL_RATIO


def _lag(first: Samples, second: Samples, max_lag: int) -> int:
    # Сдвиг второго канала относительно первого: second[t + lag] похож на first[t]
    frame = min(max(LAG_FRAME, 4 * max_lag), len(first))
    if frame == 0:
        return 0
    starts = range(0, len(first) - frame + 1, frame)
    loud = sorted(
        starts,
        key=lambda start: float(
            np.dot(first[start : start + frame], first[start : start + frame])
        ),
    )
    totals = np.zeros(2 * max_lag + 1)
    for start in loud[-LAG_FRAMES:]:
        low, high = start - max_lag, start + frame + max_lag
        window = second[max(low, 0) : min(high, len(second))].astype(np.float64)
        window = np.pad(window, (max(-low, 0), max(high - len(second), 0)))
        totals += np.correlate(window, first[start : start + frame].astype(np.float64), "valid")
    return int(np.argmax(totals)) - max_lag


def cut_points(tracks: Sequence[Samples], sample_rate: int) -> list[int]:
    """
    Границы окон распознавания

    Описание:
        Окна идут через WINDOW_SECONDS. Граница ищется в пределах CUT_SEARCH_SECONDS
        от нее. Энергия каждого канала по кадрам CUT_FRAME_SECONDS берется в децибелах
        относительно громких кадров канала (95-й процентиль) и ограничивается
        от -CUT_FLOOR_DB до 0: постоянный шум и цифровая тишина с редкими всплесками
        не перевешивают паузу в речи другого канала. Уровни каналов суммируются
        и сглаживаются по CUT_QUIET_SECONDS; граница - середина самого тихого отрезка.
        Последнее окно может быть длиннее на CUT_SEARCH_SECONDS

    Аргументы:
        tracks (list[numpy.ndarray]): Каналы одной длины
        sample_rate (int): Частота дискретизации, Гц

    Вывод:
        list[int]: Номера отсчетов от 0 до длины записи включительно

    Пример использования:
        >>> import numpy as np
        >>> from stt_ru.pipeline import cut_points
        >>> cut_points([np.zeros(16000, dtype=np.float32)], 16000)
        [0, 16000]
    """
    total = len(tracks[0])
    step = int(WINDOW_SECONDS * sample_rate)
    search = int(CUT_SEARCH_SECONDS * sample_rate)
    frame = max(int(CUT_FRAME_SECONDS * sample_rate), 1)
    frames = max(2 * search // frame, 1)
    width = min(max(round(CUT_QUIET_SECONDS / CUT_FRAME_SECONDS), 1), frames)
    cuts = [0]
    while total - cuts[-1] > step + search:
        low = cuts[-1] + step - search
        energy = np.zeros(frames)
        for track in tracks:
            channel = np.square(track[low : low + frames * frame], dtype=np.float64)
            channel = channel.reshape(frames, frame).sum(axis=1)
            loud = np.percentile(channel, 95)
            level = 10 * np.log10((channel + 1e-20) / (loud + 1e-20))
            energy += np.clip(level, -CUT_FLOOR_DB, 0)
        quiet = np.convolve(energy, np.ones(width), mode="valid")
        cuts.append(low + (int(np.argmin(quiet)) + width // 2) * frame + frame // 2)
    cuts.append(total)
    return cuts
