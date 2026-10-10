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
# Каналы с энергией разности ниже -30 дБ - одна запись, сохраненная в несколько каналов
SAME_CHANNEL_RATIO = 1e-3

Track = tuple[int | None, Samples]


class Transcriber:
    """
    Распознавание аудиофайлов

    Описание:
        Каналы стереозаписи распознаются отдельно: в записях разговоров каждый канал - свой
        говорящий; одинаковые каналы (энергия разности ниже -30 дБ) распознаются один раз
        как сведенная запись. Файл с частотой, которую модель не принимает,
        передискретизируется в 16 кГц через ffmpeg (см. load_audio). Записи длиннее 10 минут
        распознаются окнами около 10 минут с границей в паузе (см. cut_points)

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
        try:
            native: int | None = sf.info(os.fspath(path)).samplerate
        except sf.LibsndfileError:
            native = None
        rate = RESAMPLE_RATE if native is not None and native not in SAMPLE_RATES else None
        audio = load_audio(path, sample_rate=rate, ffmpeg=self.ffmpeg)
        if audio.sample_rate not in SAMPLE_RATES:
            audio = load_audio(path, sample_rate=RESAMPLE_RATE, ffmpeg=self.ffmpeg)
        return self.transcribe_audio(audio, source=Path(path).name)

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
        tracks = self._tracks(audio)
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
            tuple(segments), audio.duration, len(tracks), self.recognizer.model, source
        )

    def _tracks(self, audio: Audio) -> list[Track]:
        if not self.split_channels:
            return [(None, audio.mono())]
        tracks: list[Track] = []
        for index in range(audio.num_channels):
            samples = audio.samples[:, index]
            if not any(same_signal(samples, kept) for _, kept in tracks):
                tracks.append((index, samples))
        if len(tracks) == 1:
            return [(None, tracks[0][1])]
        return tracks


def same_signal(first: Samples, second: Samples, block: int = 1 << 20) -> bool:
    """
    Совпадение двух каналов

    Аргументы:
        first (numpy.ndarray): Отсчеты первого канала
        second (numpy.ndarray): Отсчеты второго канала той же длины
        block (int): Число отсчетов, которые сравниваются за раз

    Вывод:
        bool: Энергия разности не больше SAME_CHANNEL_RATIO от энергии громкого канала;
            два тихих канала совпадают

    Пример использования:
        >>> import numpy as np
        >>> from stt_ru.pipeline import same_signal
        >>> signal = np.sin(np.arange(8000, dtype=np.float32))
        >>> same_signal(signal, signal + 0.01), same_signal(signal, -signal)
        (True, False)
    """
    difference = energy_first = energy_second = 0.0
    for start in range(0, len(first), block):
        a = first[start : start + block].astype(np.float64)
        b = second[start : start + block].astype(np.float64)
        difference += float(np.square(a - b).sum())
        energy_first += float(np.dot(a, a))
        energy_second += float(np.dot(b, b))
    return difference <= SAME_CHANNEL_RATIO * max(energy_first, energy_second)


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
