import warnings
from collections.abc import Callable
from numbers import Integral

import numpy as np

from ..exceptions import ParameterError

DEFAULT_THRESHOLD = 0.7
# Кластеры с меньшей долей речи - ошибки сегментации, а не отдельные люди
MIN_SHARE = 0.02
# Память на матрицу расстояний растет как квадрат числа векторов
MAX_VECTORS = 4000
# Память на реплики растет с числом говорящих, а число может прийти от клиента сервиса
MAX_SPEAKERS = 100
# Больше этого числа сверх найденного лишние говорящие заметно дробят людей
EXTRA_SPEAKERS = 3
LLOYD_STEPS = 30
# Центры ближе этого косинусного расстояния - части одного человека, а не разные люди
MIN_CENTER_DISTANCE = 0.15


def cluster(
    vectors: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    speakers: int | None = None,
    weights: np.ndarray | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Кластеризация эмбеддингов говорящих

    Описание:
        Агломеративная кластеризация: каждый раз сливаются два кластера с ближайшими центрами,
        пока расстояние между ними не больше threshold. Кластеры с долей веса меньше MIN_SHARE
        присоединяются к ближайшему большому. С точным числом кластеров вместо этого остаются
        speakers самых весомых, а если их меньше, разбиение продолжается до этого числа; затем
        векторы несколько раз перераспределяются между центрами, если от этого не пропадает
        кластер и центры не сходятся. Если векторов больше MAX_VECTORS, кластеризуются
        равномерно выбранные MAX_VECTORS, а остальные относятся к ближайшему центру.
        Результат не зависит от случайности

    Аргументы:
        vectors (numpy.ndarray): Нормированные векторы, форма (число векторов, размерность)
        threshold (float): Наибольшее евклидово расстояние между центрами сливаемых кластеров
        speakers (int): Точное число кластеров, если векторов не меньше; None - число определяется
            по threshold и MIN_SHARE
        weights (numpy.ndarray): Вес каждого вектора, например длительность речи; None или все
            нули - равные
        on_warning (Callable): Получает текст предупреждения о завышенном числе; None -
            warnings.warn

    Вывод:
        numpy.ndarray: Номер кластера каждого вектора, с нуля

    Исключения:
        ParameterError: Если число кластеров неверно (см. check_speakers)

    Пример использования:
        >>> import numpy as np
        >>> from stt_ru.diarization import cluster
        >>> vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
        >>> cluster(vectors).tolist()
        [0, 0, 1, 1]
        >>> cluster(vectors, speakers=1).tolist()
        [0, 0, 0, 0]
    """
    check_speakers(speakers)
    count = len(vectors)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    weights = np.ones(count) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(count)
    picked = np.linspace(0, count - 1, min(count, MAX_VECTORS)).astype(np.int64)
    chosen = vectors[picked].astype(np.float64)
    used = weights[picked]
    labels = _agglomerate(chosen, threshold)
    automatic = _absorb_small(chosen, labels, used, None)
    if speakers is None:
        labels = automatic
    else:
        found = len(np.unique(automatic))
        if speakers > found + EXTRA_SPEAKERS:
            text = (
                f"Указано говорящих {speakers}, а по записи их около {found}: "
                "лишние говорящие дробят людей"
            )
            if on_warning is None:
                warnings.warn(text, stacklevel=2)
            else:
                on_warning(text)
        if len(np.unique(labels)) < speakers:
            labels = _agglomerate(chosen, threshold, speakers)
        labels = _refined(chosen, _absorb_small(chosen, labels, used, speakers), used)
    if len(picked) < count:
        full = _closest(vectors, vectors[picked], labels)
        full[picked] = labels
        labels = full
    return np.unique(labels, return_inverse=True)[1]


def check_speakers(speakers: object) -> None:
    """
    Проверка числа говорящих

    Аргументы:
        speakers (int): Точное число говорящих; None - определяется по записи

    Исключения:
        ParameterError: Если это не целое число от 1 до MAX_SPEAKERS

    Пример использования:
        >>> from stt_ru.diarization import check_speakers
        >>> check_speakers(2)
        >>> check_speakers(None)
    """
    if speakers is None:
        return
    whole = isinstance(speakers, Integral) and not isinstance(speakers, bool)
    if not whole or not 1 <= int(speakers) <= MAX_SPEAKERS:  # type: ignore[call-overload]
        raise ParameterError(
            f"Число говорящих должно быть целым от 1 до {MAX_SPEAKERS}: {speakers}"
        )


def _agglomerate(vectors: np.ndarray, threshold: float, speakers: int | None = None) -> np.ndarray:
    count = len(vectors)
    centers = vectors.copy()
    sizes = np.ones(count)
    labels = np.arange(count)
    distances = np.sqrt(np.maximum(_squared(centers, centers), 0))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    alive = np.ones(count, dtype=bool)
    groups = count
    while groups > (speakers or 1):
        rows = np.flatnonzero(alive)
        kept = rows[distances[rows, nearest[rows]].argmin()]
        gone = nearest[kept]
        if speakers is None and distances[kept, gone] > threshold:
            break
        total = sizes[kept] + sizes[gone]
        centers[kept] = (sizes[kept] * centers[kept] + sizes[gone] * centers[gone]) / total
        sizes[kept] = total
        labels[labels == gone] = kept
        alive[gone] = False
        groups -= 1
        distances[gone, :] = distances[:, gone] = np.inf
        fresh = np.sqrt(np.maximum(_squared(centers[kept : kept + 1], centers), 0))[0]
        fresh[~alive] = np.inf
        fresh[kept] = np.inf
        distances[kept, :] = distances[:, kept] = fresh
        stale = np.flatnonzero(alive & ((nearest == kept) | (nearest == gone)))
        nearest[stale] = distances[stale].argmin(axis=1)
        closer = alive & (fresh < distances[np.arange(count), nearest])
        nearest[closer] = kept
        nearest[kept] = fresh.argmin()
    return labels


def _squared(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    sizes: np.ndarray = np.sum(first**2, axis=1, keepdims=True)
    products: np.ndarray = first @ second.T
    return np.asarray(sizes - 2 * products + np.sum(second**2, axis=1))


def _closest(vectors: np.ndarray, members: np.ndarray, labels: np.ndarray) -> np.ndarray:
    kept = np.unique(labels)
    centers = np.stack([members[labels == label].mean(axis=0) for label in kept])
    # По косинусу: у среднего большого кластера норма меньше, и по расстоянию он был бы ближе всех
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    closest: np.ndarray = kept[(vectors @ centers.T).argmax(axis=1)]
    return closest


def _refined(vectors: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Если при пороге двое слиплись, второй кластер мелкий: перераспределение возвращает ему людей
    scale = np.maximum(weights, 1e-6)[:, None]
    refined = labels
    for _ in range(LLOYD_STEPS):
        kept, centers = _weighted_centers(vectors, refined, scale)
        moved = kept[(vectors @ centers.T).argmax(axis=1)]
        if np.array_equal(moved, refined):
            break
        refined = moved
    kept, centers = _weighted_centers(vectors, refined, scale)
    # Мелкий кластер рядом с крупным мог забрать половину речи того же человека
    similar = centers @ centers.T
    np.fill_diagonal(similar, -1.0)
    if len(kept) < len(np.unique(labels)) or 1 - similar.max() < MIN_CENTER_DISTANCE:
        return labels
    return refined


def _weighted_centers(
    vectors: np.ndarray, labels: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    kept = np.unique(labels)
    centers = np.stack([np.sum((vectors * scale)[labels == label], axis=0) for label in kept])
    return kept, centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)


def _absorb_small(
    vectors: np.ndarray, labels: np.ndarray, weights: np.ndarray, speakers: int | None
) -> np.ndarray:
    uniques, inverse = np.unique(labels, return_inverse=True)
    totals = np.bincount(inverse, weights=weights)
    if speakers is None:
        kept = uniques[totals >= MIN_SHARE * totals.sum()]
        if len(kept) == 0:
            kept = uniques[[totals.argmax()]]
    else:
        kept = uniques[np.argsort(-totals, kind="stable")[:speakers]]
    if len(kept) == len(uniques):
        return labels
    large = np.isin(labels, kept)
    result = labels.copy()
    result[~large] = _closest(vectors[~large], vectors[large], labels[large])
    return result
