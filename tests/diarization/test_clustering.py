import numpy as np
import pytest

from stt_ru.diarization import cluster, clustering
from stt_ru.exceptions import ParameterError


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


def test_outlier_does_not_take_place_of_speaker():
    # Если просто остановиться на двух кластерах, второй - это выброс, а люди сливаются
    vectors = np.concatenate([groups(10, 10), [[-0.8, -0.6]]])
    labels = cluster(vectors, speakers=2)
    assert labels[:10].tolist() == [0] * 10
    assert labels[10:20].tolist() == [1] * 10
    assert labels[20] in (0, 1)


def test_rare_speaker_found_with_exact_speakers():
    vectors = np.concatenate([groups(40), around(2.0, 1)])
    weights = np.array([100.0] * 40 + [1.0])
    assert len(set(cluster(vectors, weights=weights).tolist())) == 1
    assert cluster(vectors, speakers=2, weights=weights).tolist() == [0] * 40 + [1]


def test_fewer_speakers_keep_largest():
    labels = cluster(groups(9, 2, 5), speakers=2)
    assert labels[:9].tolist() == [0] * 9
    assert labels[11:].tolist() == [1] * 5


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
    assert cluster(groups(30, 30), speakers=2).tolist() == [0] * 30 + [1] * 30


def test_stable():
    vectors = groups(9, 9, 9)
    assert cluster(vectors).tolist() == cluster(vectors).tolist()


@pytest.mark.parametrize("count", [0, 1])
def test_few_vectors(count):
    assert cluster(np.ones((count, 2)) / np.sqrt(2)).tolist() == [0] * count


def test_loud_cluster_without_speech_does_not_take_place():
    # Музыка: много времени, но почти нет распознанной речи; вес считается только по речи
    vectors = np.concatenate([groups(10, 10), around(3.0, 30, seed=9)])
    weights = np.array([100.0] * 10 + [100.0] * 10 + [0.0] * 30)
    labels = cluster(vectors, speakers=2, weights=weights)
    assert len(set(labels[:10].tolist())) == 1
    assert len(set(labels[10:20].tolist())) == 1
    assert labels[0] != labels[10]


def test_stuck_voices_separated_when_refined():
    # При пороге почти все слиплось, а от второго человека отделился только маленький кусок
    first, second = around(0.0, 40, spread=0.05, seed=1), around(1.0, 40, spread=0.05, seed=2)
    vectors = np.concatenate([first, second])
    stuck = np.array([0] * 40 + [0] * 35 + [1] * 5)
    labels = clustering._refined(vectors, stuck, np.ones(80))
    assert labels.tolist() == [0] * 40 + [1] * 40


def test_refining_limited(monkeypatch):
    monkeypatch.setattr(clustering, "LLOYD_STEPS", 1)
    first, second = around(0.0, 40, spread=0.05, seed=1), around(1.0, 40, spread=0.05, seed=2)
    stuck = np.array([0] * 40 + [0] * 35 + [1] * 5)
    labels = clustering._refined(np.concatenate([first, second]), stuck, np.ones(80))
    assert labels[:5].tolist() == [0] * 5


def test_refining_keeps_stable_split():
    vectors = groups(10, 10)
    labels = np.array([0] * 10 + [1] * 10)
    assert clustering._refined(vectors, labels, np.ones(20)).tolist() == labels.tolist()


def test_refining_does_not_split_one_person():
    # Без проверки центров мелкий кластер забирает половину человека, чей голос звучит по-разному
    near = [[np.cos(a), np.sin(a)] for a in np.radians([14.0, -14.0, 36.0])]
    vectors = np.array([near[0]] * 15 + [near[1]] * 65 + [near[2]])
    labels = np.array([0] * 80 + [1])
    assert clustering._refined(vectors, labels, np.ones(81)).tolist() == labels.tolist()


def test_refining_keeps_cluster():
    # Первый кластер целиком уходит к соседям, а говорящих должно остаться столько же
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.44], [0.44, 0.9]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 2])
    assert clustering._refined(vectors, labels, np.ones(4)).tolist() == labels.tolist()


def test_zero_weights_are_equal():
    assert cluster(groups(5, 7), weights=np.zeros(12)).tolist() == [0] * 5 + [1] * 7


def test_too_many_speakers_warned():
    with pytest.warns(UserWarning, match="Указано говорящих 8, а по записи их около 2"):
        cluster(groups(10, 10), speakers=8)


def test_warning_goes_to_callback(recwarn):
    seen = []
    cluster(groups(10, 10), speakers=8, on_warning=seen.append)
    assert "Указано говорящих 8" in seen[0]
    assert len(recwarn) == 0


@pytest.mark.parametrize("speakers", [2, 5])
def test_near_number_not_warned(speakers, recwarn):
    cluster(groups(10, 10), speakers=speakers)
    assert not recwarn.list


@pytest.mark.parametrize("speakers", [0, -1, 101, 2.5, 2.0, True, "2", [2]])
def test_wrong_number(speakers):
    with pytest.raises(ParameterError, match="Число говорящих должно быть целым от 1 до 100"):
        cluster(groups(3, 3), speakers=speakers)


def test_numpy_integer_is_fine():
    assert len(set(cluster(groups(3, 3), speakers=np.int64(2)).tolist())) == 2
