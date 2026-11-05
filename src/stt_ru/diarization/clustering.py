import numpy as np

DEFAULT_THRESHOLD = 0.7
# Кластеры с меньшей долей речи - ошибки сегментации, а не отдельные люди
MIN_SHARE = 0.02
# Память на матрицу расстояний растет как квадрат числа векторов
MAX_VECTORS = 4000


def cluster(
    vectors: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    speakers: int | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Кластеризация эмбеддингов говорящих

    Описание:
        Агломеративная кластеризация: каждый раз сливаются два кластера с ближайшими центрами,
        пока расстояние между ними не больше threshold. Кластеры с долей веса меньше MIN_SHARE
        присоединяются к ближайшему большому. Если векторов больше MAX_VECTORS, кластеризуются
        равномерно выбранные MAX_VECTORS, а остальные относятся к ближайшему центру.
        Результат не зависит от случайности

    Аргументы:
        vectors (numpy.ndarray): Нормированные векторы, форма (число векторов, размерность)
        threshold (float): Наибольшее евклидово расстояние между центрами сливаемых кластеров
        speakers (int): Точное число кластеров: threshold и MIN_SHARE не применяются;
            None - число определяется по threshold
        weights (numpy.ndarray): Вес каждого вектора, например длительность речи; None - равные

    Вывод:
        numpy.ndarray: Номер кластера каждого вектора, с нуля

    Пример использования:
        >>> import numpy as np
        >>> from stt_ru.diarization import cluster
        >>> vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
        >>> cluster(vectors).tolist()
        [0, 0, 1, 1]
        >>> cluster(vectors, speakers=1).tolist()
        [0, 0, 0, 0]
    """
    count = len(vectors)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    weights = np.ones(count) if weights is None else np.asarray(weights, dtype=np.float64)
    picked = np.linspace(0, count - 1, min(count, MAX_VECTORS)).astype(np.int64)
    labels = _agglomerate(vectors[picked].astype(np.float64), threshold, speakers)
    if speakers is None:
        labels = _absorb_small(vectors[picked], labels, weights[picked])
    if len(picked) < count:
        full = _closest(vectors, vectors[picked], labels)
        full[picked] = labels
        labels = full
    return np.unique(labels, return_inverse=True)[1]


def _agglomerate(vectors: np.ndarray, threshold: float, speakers: int | None) -> np.ndarray:
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


def _absorb_small(vectors: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    uniques, inverse = np.unique(labels, return_inverse=True)
    shares = np.bincount(inverse, weights=weights) / max(weights.sum(), 1e-9)
    kept = uniques[shares >= MIN_SHARE]
    if len(kept) == len(uniques):
        return labels
    if len(kept) == 0:
        kept = uniques[[shares.argmax()]]
    large = np.isin(labels, kept)
    result = labels.copy()
    result[~large] = _closest(vectors[~large], vectors[large], labels[large])
    return result
