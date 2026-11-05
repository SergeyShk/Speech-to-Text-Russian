import contextlib
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import onnxruntime as rt
from onnx_asr.loader import Manager, create_vad_resolver
from onnx_asr.utils import SampleRates

from ..asr.recognizer import SAMPLE_RATES, default_providers
from ..audio.loader import Samples
from ..exceptions import AudioError, ModelError, ParameterError
from .clustering import DEFAULT_THRESHOLD, cluster

EMBEDDING_MODEL = "Wespeaker/wespeaker-voxceleb-campplus-LM"
SEGMENTATION_MODEL = "onnx-community/pyannote-segmentation-3.0"
SAMPLE_RATE = 16000
WINDOW = 10 * SAMPLE_RATE
HOP = int(2.5 * SAMPLE_RATE)
# Шаг и поле зрения выхода модели сегментации в отсчетах
FRAME_STEP = 270
FRAME_FIELD = 991
# Выход модели - вероятности наборов: никто, 1, 2, 3, 1 и 2, 1 и 3, 2 и 3
POWERSET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int8
)
MIN_ACTIVE = int(0.1 * SAMPLE_RATE)
MIN_CLEAN = int(0.4 * SAMPLE_RATE)
MIN_CHUNK = int(0.5 * SAMPLE_RATE)
# onnx-asr передискретизирует запись целиком: части ограничивают память
RESAMPLE_BLOCK = 600
# Память на реплики растет с числом говорящих, а число может прийти от клиента сервиса
MAX_SPEAKERS = 100


def check_speakers(speakers: int | None) -> None:
    """
    Проверка числа говорящих

    Аргументы:
        speakers (int): Точное число говорящих; None - определяется по записи

    Исключения:
        ParameterError: Если число не от 1 до MAX_SPEAKERS
    """
    if speakers is not None and not 1 <= speakers <= MAX_SPEAKERS:
        raise ParameterError(f"Число говорящих должно быть от 1 до {MAX_SPEAKERS}: {speakers}")


@dataclass(frozen=True)
class Turn:
    """
    Реплика: отрезок, где говорит один человек

    Аргументы:
        start (float): Начало, с
        end (float): Конец, с
        speaker (int): Номер говорящего, с единицы, по порядку появления в записи

    Пример использования:
        >>> from stt_ru.diarization import Turn
        >>> Turn(0.5, 2.0, 1)
        Turn(start=0.5, end=2.0, speaker=1)
    """

    start: float
    end: float
    speaker: int


class Diarizer:
    """
    Разделение записи на говорящих

    Описание:
        В окнах 10 с модель сегментации находит до трех говорящих, по их речи модель эмбеддингов
        считает голосовые отпечатки, а кластеризация (см. cluster) сводит их в людей. Реплики
        разных людей могут перекрываться. При первом использовании модели скачиваются
        с Hugging Face в его кэш, дальше загружаются из кэша

    Аргументы:
        threshold (float): Порог кластеризации: чем больше, тем меньше говорящих (см. cluster)
        providers (list[str]): Провайдеры ONNX Runtime; None - default_providers()

    Исключения:
        ModelError: Если модели не удалось загрузить
        ParameterError: Если порог не положительное конечное число

    Пример использования:
        >>> from stt_ru.diarization import Diarizer
        >>> turns = Diarizer().diarize(samples, 16000)  # doctest: +SKIP
        >>> turns[0]  # doctest: +SKIP
        Turn(start=0.98, end=9.05, speaker=1)
    """

    def __init__(
        self, *, threshold: float = DEFAULT_THRESHOLD, providers: Sequence[str] | None = None
    ) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ParameterError(f"threshold должен быть положительным числом: {threshold}")
        self.threshold = threshold
        chosen = default_providers() if providers is None else list(providers)
        options = rt.SessionOptions()
        options.log_severity_level = 3
        try:
            # ONNX Runtime печатает в stdout ошибку провайдера, а stdout - для результата
            with contextlib.redirect_stdout(sys.stderr):
                self._embedder = Manager(options, chosen).create_se(EMBEDDING_MODEL)
                files = create_vad_resolver(SEGMENTATION_MODEL).resolve_model()
                self._segmenter = rt.InferenceSession(
                    files["model"], sess_options=options, providers=chosen
                )
        except Exception as error:
            raise ModelError(f"Не удалось загрузить модели диаризации: {error}") from error

    def diarize(
        self, samples: Samples, sample_rate: int, speakers: int | None = None
    ) -> list[Turn]:
        """
        Поиск реплик в одноканальной записи

        Аргументы:
            samples (numpy.ndarray): Одномерный массив отсчетов float32
            sample_rate (int): Частота дискретизации, Гц
            speakers (int): Точное число говорящих; None - определяется по записи

        Вывод:
            list[Turn]: Реплики по времени начала; пусто, если речи нет

        Исключения:
            AudioError: Если отсчеты не одномерный массив float32 или частота не поддерживается
            ParameterError: Если число говорящих неверно (см. check_speakers)
        """
        check_speakers(speakers)
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise AudioError(
                f"Отсчеты должны быть одномерным массивом float32: {samples.dtype}, "
                f"форма {samples.shape}"
            )
        waveform = self._resample(samples, sample_rate)
        starts, active = self._segment(waveform)
        vectors, owners, weights = [], [], []
        # Куски считаются сразу: вместе они занимали бы в несколько раз больше самой записи.
        # В пачке короткие куски разбавлялись бы нулями: onnx-asr не передает модели их длины
        for chunk, owner, weight in _local_speakers(waveform, starts, active):
            vectors.append(self._embedder.embedding(chunk))
            owners.append(owner)
            weights.append(weight)
        if not vectors:
            return []
        matrix = np.stack(vectors)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        labels = cluster(matrix, self.threshold, speakers, np.array(weights))
        return _turns(len(waveform), starts, active, owners, labels)

    def _resample(self, samples: Samples, sample_rate: int) -> Samples:
        # Один NaN или inf портит все окно, куда он попал
        if not np.isfinite(samples).all():
            samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        if sample_rate == SAMPLE_RATE:
            return samples
        if sample_rate not in SAMPLE_RATES:
            raise AudioError(f"Частота {sample_rate} Гц не поддерживается")
        block = RESAMPLE_BLOCK * sample_rate
        parts = []
        for start in range(0, len(samples), block):
            part = samples[start : start + block][None, :]
            resampled, lengths = self._embedder.resampler(
                part, np.array([part.shape[1]], dtype=np.int64), cast(SampleRates, sample_rate)
            )
            parts.append(resampled[0, : int(lengths[0])])
        return np.concatenate(parts) if parts else samples[:0]

    def _segment(self, waveform: Samples) -> tuple[list[int], np.ndarray]:
        total = len(waveform)
        starts = list(range(0, max(total - WINDOW, 0) + 1, HOP)) if total else []
        if total > WINDOW and starts[-1] + WINDOW < total:
            starts.append(total - WINDOW)
        windows = []
        for start in starts:
            chunk = waveform[start : start + WINDOW]
            if len(chunk) < WINDOW:
                chunk = np.pad(chunk, (0, WINDOW - len(chunk)))
            logits = self._segmenter.run(["logits"], {"input_values": chunk[None, None, :]})[0]
            windows.append(POWERSET[logits[0].argmax(axis=-1)])
        return starts, np.stack(windows) if windows else np.zeros((0, 0, 3), dtype=np.int8)


def _local_speakers(
    waveform: Samples, starts: list[int], active: np.ndarray
) -> Iterator[tuple[Samples, tuple[int, int], int]]:
    alone = active.sum(axis=-1) == 1
    for window, start in enumerate(starts):
        piece = waveform[start : start + WINDOW]
        for speaker in range(active.shape[-1]):
            voiced = active[window, :, speaker] > 0
            if voiced.sum() * FRAME_STEP < MIN_ACTIVE:
                continue
            clean = voiced & alone[window]
            chosen = clean if clean.sum() * FRAME_STEP >= MIN_CLEAN else voiced
            mask = np.repeat(chosen, FRAME_STEP)[: len(piece)]
            chunk = piece[: len(mask)][mask]
            if len(chunk) > 0:
                stretched = chunk if len(chunk) >= MIN_CHUNK else np.resize(chunk, MIN_CHUNK)
                yield stretched, (window, speaker), len(chunk)


def _turns(
    total: int,
    starts: list[int],
    active: np.ndarray,
    owners: list[tuple[int, int]],
    labels: np.ndarray,
) -> list[Turn]:
    size = active.shape[1]
    count = int(labels.max()) + 1
    frames = total // FRAME_STEP + size + 1
    scores = np.zeros((count, frames), dtype=np.float32)
    windows = np.zeros(frames, dtype=np.float32)
    voices = np.zeros(frames, dtype=np.float32)
    offsets = [start // FRAME_STEP for start in starts]
    per_window = active.sum(axis=-1)
    for window, offset in enumerate(offsets):
        windows[offset : offset + size] += 1
        voices[offset : offset + size] += per_window[window]
    # Центр окна надежнее краев; при равных весах победителя определяла бы нумерация кластеров
    taper = np.hamming(size) + 0.01
    for (window, speaker), label in zip(owners, labels, strict=True):
        scores[label, offsets[window] : offsets[window] + size] += (
            active[window, :, speaker] * taper
        )
    speaking = np.rint(voices / np.maximum(windows, 1)).astype(np.int64)
    order = np.argsort(-scores, axis=0)
    binary = np.zeros_like(scores, dtype=bool)
    for frame in np.flatnonzero(speaking):
        binary[order[: speaking[frame], frame], frame] = True
    step = FRAME_STEP / SAMPLE_RATE
    shift = (FRAME_FIELD - FRAME_STEP) / 2 / SAMPLE_RATE
    duration = total / SAMPLE_RATE
    found = []
    for label in range(count):
        edges = np.flatnonzero(np.diff(binary[label].astype(np.int8), prepend=0, append=0))
        found += [
            (float(first) * step + shift, min(float(last) * step + shift, duration), label)
            for first, last in zip(edges[::2], edges[1::2], strict=True)
        ]
    found = [item for item in found if item[0] < item[1]]
    found.sort()
    numbers: dict[int, int] = {}
    return [
        Turn(round(start, 3), round(end, 3), numbers.setdefault(label, len(numbers) + 1))
        for start, end, label in found
    ]
