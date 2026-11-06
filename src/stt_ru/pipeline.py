import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from warnings import warn

import numpy as np
import soundfile as sf

from .asr import DEFAULT_MODEL, DEFAULT_QUANTIZATION, SAMPLE_RATES, Recognizer
from .asr.recognizer import Word
from .audio import Audio, load_audio
from .audio.loader import Samples
from .diarization import Diarizer, Timeline, check_speakers
from .exceptions import ModelError, ParameterError, StoppedError
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
# Настоящая стереозапись одного источника: волна разная, а громкость меняется одинаково.
# На подкастах корреляция от 0.98, на звонке с говорящими по каналам она около нуля
SAME_ENVELOPE = 0.95
ENVELOPE_FRAME = 0.1
ENVELOPE_MIN_FRAMES = 50
# Канал тише громкого на 50 дБ по энергии: цифровая тишина или шум без речи
SILENT_CHANNEL = 1e-5
# Первая задача с разделением ждет модели не дольше этого; скачивание идет дальше в фоне
LOAD_TIMEOUT = 300.0

Track = tuple[int | None, Samples]
Fragment = tuple[float, float, str, list[Word] | None]


class Transcriber:
    """
    Распознавание аудиофайлов

    Описание:
        Каналы стереозаписи распознаются отдельно: в записях разговоров каждый канал - свой
        говорящий. Каналы, которые отличаются только громкостью и сдвигом до 10 мс
        (см. same_signal), распознаются один раз; каналы одного источника с разной волной
        (см. same_envelope) сводятся в один, а канал без звука пропускается. Файл с частотой, которую модель
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
        diarize (bool): Разделять говорящих в записи с одной дорожкой (см. Diarizer). Фрагмент
            со сменой говорящего делится по словам; модель, которая не сообщает время слов,
            отдает фрагмент целиком тому, кто говорит в нем дольше всех. Разные каналы
            и так разделены: для них диаризация не выполняется, нужен split_channels=False
        speakers (int): Точное число говорящих в записи, включает diarize;
            None - определяется по записи. Параметры transcribe и transcribe_audio заменяют
            diarize и speakers для одной записи (см. speaker_options); модели диаризации
            при этом загружаются при первом использовании

    Исключения:
        ModelError: Если модель не загружается
        ParameterError: Если speakers не от 1 до MAX_SPEAKERS

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
        diarize: bool = False,
        speakers: int | None = None,
    ) -> None:
        check_speakers(speakers)
        self.recognizer = Recognizer(
            model, quantization=quantization, model_dir=model_dir, providers=providers, vad=vad
        )
        self.split_channels = split_channels
        self.ffmpeg = ffmpeg
        self.speakers = speakers
        self.diarize = diarize or speakers is not None
        self.diarizer = Diarizer(providers=providers) if self.diarize else None
        self._providers = providers
        self._diarizer_lock = threading.Lock()
        self._loader: threading.Thread | None = None
        self._load_error: ModelError | None = None
        self._load_deadline = 0.0

    def speaker_options(
        self, diarize: bool | None = None, speakers: int | None = None
    ) -> tuple[bool, int | None]:
        """
        Параметры разделения говорящих для одной записи

        Описание:
            speakers включает разделение; None берет значение, заданное при создании

        Аргументы:
            diarize (bool): Разделять говорящих; None - как задано при создании
            speakers (int): Точное число говорящих; None - как задано при создании

        Вывод:
            tuple[bool, int | None]: Разделять ли говорящих и их точное число, если известно

        Исключения:
            ParameterError: Если speakers не от 1 до MAX_SPEAKERS или задан вместе с diarize=False
        """
        check_speakers(speakers)
        if diarize is False:
            if speakers is not None:
                raise ParameterError("Число говорящих задано, а разделение говорящих выключено")
            return False, None
        if speakers is not None:
            return True, speakers
        return (True, self.speakers) if diarize or self.diarize else (False, None)

    def transcribe(
        self,
        path: str | os.PathLike[str],
        *,
        diarize: bool | None = None,
        speakers: int | None = None,
    ) -> Transcript:
        """
        Распознавание аудиофайла

        Аргументы:
            path (str): Путь к файлу (см. load_audio)
            diarize (bool): Разделять говорящих (см. speaker_options)
            speakers (int): Точное число говорящих (см. speaker_options)

        Вывод:
            Transcript: Фрагменты речи по времени начала

        Исключения:
            FileNotFoundError: Если файла нет
            AudioError: Если файл не читается
            ModelError: Если не загружаются модели диаризации
            ParameterError: Если параметры говорящих неверны (см. speaker_options)
        """
        self.speaker_options(diarize, speakers)
        audio = self.load(path)
        return self.transcribe_audio(
            audio, source=Path(path).name, diarize=diarize, speakers=speakers
        )

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

    def transcribe_audio(
        self,
        audio: Audio,
        source: str | None = None,
        *,
        diarize: bool | None = None,
        speakers: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Transcript:
        """
        Распознавание записи в памяти

        Аргументы:
            audio (Audio): Запись с частотой из SAMPLE_RATES
            source (str): Имя файла записи для результата
            diarize (bool): Разделять говорящих (см. speaker_options)
            speakers (int): Точное число говорящих (см. speaker_options)
            should_stop (Callable): Вызывается между шагами; True прерывает работу

        Вывод:
            Transcript: Фрагменты речи по времени начала; фрагменты без текста отбрасываются

        Исключения:
            AudioError: Если модель не принимает частоту записи
            ModelError: Если не загружаются модели диаризации
            ParameterError: Если параметры говорящих неверны (см. speaker_options)
            StoppedError: Если should_stop вернул True
        """
        separate, count = self.speaker_options(diarize, speakers)
        tracks, channel_map = self._tracks(audio)
        diarizer = self._diarizer_for(tracks, source, should_stop) if separate else None
        cuts = cut_points([samples for _, samples in tracks], audio.sample_rate)
        segments: list[Segment] = []
        fragments: list[Fragment] = []
        for start, end in pairwise(cuts):
            if end == start:
                continue
            _check_stop(should_stop)
            offset = start / audio.sample_rate
            waveforms = [samples[start:end] for _, samples in tracks]
            if diarizer is not None:
                found = self.recognizer.recognize_words(waveforms, audio.sample_rate)[0]
                fragments += [_shifted(fragment, offset) for fragment in found if fragment[2]]
            else:
                segments += [
                    Segment(round(offset + begin, 3), round(offset + finish, 3), text, channel)
                    for (channel, _), window in zip(
                        tracks,
                        self.recognizer.recognize(waveforms, audio.sample_rate),
                        strict=True,
                    )
                    for begin, finish, text in window
                    if text
                ]
        if diarizer is not None and fragments:
            # Вес голоса считается только по распознанной речи: музыка без текста не займет
            # место человека, поэтому распознавание идет раньше
            turns = diarizer.diarize(
                tracks[0][1],
                audio.sample_rate,
                count,
                speech=[(begin, finish) for begin, finish, _, _ in fragments],
                should_stop=should_stop,
            )
            segments = _renumbered(_by_speaker(fragments, Timeline(turns)))
        segments.sort(key=lambda segment: (segment.start, segment.channel or 0))
        return Transcript(
            tuple(segments),
            audio.duration,
            len(tracks),
            self.recognizer.model,
            source,
            channel_map,
        )

    def _diarizer_for(
        self, tracks: list[Track], source: str | None, should_stop: Callable[[], bool] | None
    ) -> Diarizer | None:
        if len(tracks) > 1:
            # Python показывает повтор одного текста один раз: имя файла отличает записи
            warn(
                f"{source or 'Запись'}: каналы разные, каждый считается говорящим; "
                "чтобы разделить говорящих в сведенной записи, нужно сводить каналы в один",
                stacklevel=3,
            )
            return None
        return self._loaded_diarizer(should_stop)

    def _loaded_diarizer(self, should_stop: Callable[[], bool] | None = None) -> Diarizer:
        with self._diarizer_lock:
            if self.diarizer is None and self._loader is None:
                self._load_deadline = time.monotonic() + LOAD_TIMEOUT
                self._load_error = None
                self._loader = threading.Thread(target=self._load_diarizer, daemon=True)
                self._loader.start()
            loader = self._loader
        # Поток, а не вызов здесь: huggingface_hub ждет молчащую сеть без тайм-аута
        while loader is not None and loader.is_alive() and time.monotonic() < self._load_deadline:
            _check_stop(should_stop)
            loader.join(0.2)
        return self._finished_diarizer(loader)

    def _finished_diarizer(self, loader: threading.Thread | None) -> Diarizer:
        with self._diarizer_lock:
            if self.diarizer is not None:
                return self.diarizer
            if loader is not None and loader.is_alive():
                raise ModelError(
                    f"Модели разделения говорящих не загрузились за {LOAD_TIMEOUT:.0f} с: "
                    "проверьте сеть; загрузка продолжается"
                )
            self._loader = None
            raise self._load_error or ModelError("Модели разделения говорящих не загрузились")

    def _load_diarizer(self) -> None:
        try:
            diarizer = Diarizer(providers=self._providers)
        except ModelError as error:
            self._load_error = error
        else:
            with self._diarizer_lock:
                self.diarizer = diarizer

    def _tracks(self, audio: Audio) -> tuple[list[Track], tuple[int | None, ...]]:
        single: tuple[int | None, ...] = (None,) * audio.num_channels
        if not self.split_channels:
            return [(None, audio.mono())], single
        lag = int(SAME_CHANNEL_LAG * audio.sample_rate)
        frame = max(int(ENVELOPE_FRAME * audio.sample_rate), 1)
        energies = [_energy(audio.samples[:, index]) for index in range(audio.num_channels)]
        loudest = max(energies)
        tracks: list[Track] = []
        channel_map: list[int | None] = []
        mixed = False
        for index in range(audio.num_channels):
            if energies[index] < SILENT_CHANNEL * loudest:
                channel_map.append(index)
                continue
            samples = audio.samples[:, index]
            match = next(
                (channel for channel, kept in tracks if same_signal(kept, samples, lag)), None
            )
            if match is None:
                match = next(
                    (channel for channel, kept in tracks if same_envelope(kept, samples, frame)),
                    None,
                )
                mixed = mixed or match is not None
            if match is None:
                tracks.append((index, samples))
                channel_map.append(index)
            else:
                channel_map.append(match)
        if len(tracks) == 1:
            return [(None, audio.mono() if mixed else tracks[0][1])], single
        return tracks, tuple(channel_map)


def _check_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise StoppedError("Распознавание остановлено")


def _shifted(fragment: Fragment, offset: float) -> Fragment:
    begin, finish, text, words = fragment
    moved = None if words is None else [(a + offset, b + offset, word) for a, b, word in words]
    return begin + offset, finish + offset, text, moved


def _by_speaker(fragments: list[Fragment], timeline: Timeline) -> list[Segment]:
    segments = []
    for begin, finish, text, words in fragments:
        if not words:
            speaker = timeline.speaker(begin, finish) or 1
            segments.append(Segment(round(begin, 3), round(finish, 3), text, speaker_id=speaker))
            continue
        groups: list[tuple[int, float, list[str]]] = []
        for first, last, word in words:
            speaker = timeline.speaker(first, last) or 1
            if not groups or groups[-1][0] != speaker:
                groups.append((speaker, first, []))
            groups[-1][2].append(word)
        starts = [begin, *(group[1] for group in groups[1:])]
        ends = [*starts[1:], finish]
        segments += [
            Segment(round(start, 3), round(end, 3), " ".join(group[2]), speaker_id=group[0])
            for start, end, group in zip(starts, ends, groups, strict=True)
        ]
    return segments


def _renumbered(segments: list[Segment]) -> list[Segment]:
    # У говорящего без единого слова нет строки в результате: номера идут без пропусков
    numbers: dict[int | None, int] = {}
    return [
        replace(segment, speaker_id=numbers.setdefault(segment.speaker_id, len(numbers) + 1))
        for segment in segments
    ]


def _energy(samples: Samples, block: int = 1 << 20) -> float:
    total = 0.0
    for start in range(0, len(samples), block):
        part = samples[start : start + block].astype(np.float64)
        total += float(np.dot(part, part))
    return total


def _envelope(samples: Samples, frame: int) -> np.ndarray:
    count = len(samples) // frame
    levels = np.empty(count)
    for index in range(count):
        part = samples[index * frame : (index + 1) * frame].astype(np.float64)
        levels[index] = np.sqrt(np.dot(part, part) / frame)
    return levels


def same_envelope(first: Samples, second: Samples, frame: int) -> bool:
    """
    Один ли источник в двух каналах с разной волной

    Описание:
        Каналы одного разговора в стереозаписи отличаются волной (обработка, MP3), но громкость
        в них меняется одинаково. Источник один, если корреляция громкости по кадрам не ниже
        SAME_ENVELOPE. На записях короче ENVELOPE_MIN_FRAMES кадров всегда False: случайное
        совпадение слишком вероятно

    Аргументы:
        first (numpy.ndarray): Отсчеты первого канала
        second (numpy.ndarray): Отсчеты второго канала той же длины
        frame (int): Длина кадра, отсчеты

    Вывод:
        bool: Источник один

    Пример использования:
        >>> import numpy as np
        >>> from stt_ru.pipeline import same_envelope
        >>> rng = np.random.default_rng(0)
        >>> level = np.repeat(rng.uniform(0, 1, 60), 800)
        >>> left = (level * rng.normal(0, 1, level.size)).astype(np.float32)
        >>> right = (level * rng.normal(0, 1, level.size)).astype(np.float32)
        >>> same_envelope(left, right, 800), same_envelope(left, right[::-1], 800)
        (True, False)
    """
    if len(first) // frame < ENVELOPE_MIN_FRAMES:
        return False
    a, b = _envelope(first, frame), _envelope(second, frame)
    if a.std() == 0 or b.std() == 0:
        return False
    return float(np.corrcoef(a, b)[0, 1]) >= SAME_ENVELOPE


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
