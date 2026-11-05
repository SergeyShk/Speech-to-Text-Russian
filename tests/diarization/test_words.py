import pytest

from stt_ru.diarization import Timeline, Turn

TURNS = [Turn(0.0, 2.0, 1), Turn(2.0, 5.0, 2), Turn(4.0, 4.5, 3), Turn(8.0, 9.0, 1)]


@pytest.mark.parametrize(
    ("start", "end", "speaker"),
    [
        (0.5, 1.0, 1),
        (1.5, 2.2, 1),
        (1.5, 3.0, 2),
        (4.1, 4.4, 2),
        (5.2, 5.5, 2),
        (7.5, 7.8, 1),
        (9.5, 9.9, 1),
    ],
)
def test_speaker(start, end, speaker):
    assert Timeline(TURNS).speaker(start, end) == speaker


def test_finished_turn_does_not_subtract():
    timeline = Timeline([Turn(0.0, 10.0, 1), Turn(1.0, 2.0, 2), Turn(9.0, 12.0, 2)])
    assert timeline.speaker(9.5, 11.0) == 2


def test_before_first_turn():
    assert Timeline([Turn(3.0, 4.0, 2)]).speaker(0.0, 1.0) == 2


def test_turn_inside_longer_one():
    timeline = Timeline([Turn(0.0, 10.0, 1), Turn(1.0, 2.0, 2)])
    assert timeline.speaker(5.0, 6.0) == 1
    assert timeline.speaker(10.5, 11.0) == 1


def test_no_turns():
    assert Timeline([]).speaker(0.0, 1.0) is None
