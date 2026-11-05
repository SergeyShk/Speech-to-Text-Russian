import numpy as np
import pytest

from stt_ru.diarization import cluster, clustering


def around(angle, count, spread=0.02, seed=0):
    rng = np.random.default_rng(seed)
    angles = angle + rng.normal(0, spread, count)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def groups(*sizes):
    return np.concatenate(
        [around(1.2 * index, size, seed=index) for index, size in enumerate(sizes)]
    )


def test_two_groups():
    labels = cluster(groups(5, 7))
    assert labels.tolist() == [0] * 5 + [1] * 7


def test_threshold():
    vectors = groups(5, 7)
    assert len(set(cluster(vectors, threshold=5.0).tolist())) == 1
    assert len(set(cluster(vectors, threshold=1e-6).tolist())) > 2


@pytest.mark.parametrize(("speakers", "found"), [(1, 1), (2, 2), (3, 3), (100, 12)])
def test_exact_speakers(speakers, found):
    assert len(set(cluster(groups(5, 7), speakers=speakers).tolist())) == found


def test_centers_merged_by_size():
    # Центр большой группы ближе к ней: одиночный вектор сливается только через порог
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.6, 0.8]])
    assert cluster(vectors, threshold=0.5).tolist() == [0, 0, 0, 1]
    assert cluster(vectors, threshold=0.95).tolist() == [0, 0, 0, 0]


def test_small_cluster_absorbed_by_share():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.7, 0.71]])
    weights = np.array([50.0, 50.0, 50.0, 50.0, 1.0])
    labels = cluster(vectors, threshold=0.2, weights=weights)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    assert labels[4] in (labels[0], labels[2])
    assert len(set(labels.tolist())) == 2


def test_small_cluster_kept_with_exact_speakers():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = cluster(vectors, speakers=2, weights=np.array([500.0, 500.0, 1.0]))
    assert labels.tolist() == [0, 0, 1]


def test_only_small_clusters(monkeypatch):
    monkeypatch.setattr(clustering, "MIN_SHARE", 0.9)
    assert set(cluster(groups(3, 3), threshold=0.1).tolist()) == {0}


def test_many_vectors(monkeypatch):
    monkeypatch.setattr(clustering, "MAX_VECTORS", 10)
    labels = cluster(groups(30, 30))
    assert labels.tolist() == [0] * 30 + [1] * 30


def test_stable():
    vectors = groups(9, 9, 9)
    assert cluster(vectors).tolist() == cluster(vectors).tolist()


@pytest.mark.parametrize("count", [0, 1])
def test_few_vectors(count):
    assert cluster(np.ones((count, 2)) / np.sqrt(2)).tolist() == [0] * count
