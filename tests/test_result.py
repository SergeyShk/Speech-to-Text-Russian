import dataclasses

import pytest

from stt_ru import Segment, Transcript


def test_segment_speaker():
    assert Segment(0.0, 1.0, "Да.", channel=0).speaker == "Канал 0"
    assert Segment(0.0, 1.0, "Да.").speaker is None


def test_segment_duration():
    assert Segment(0.994, 9.054, "Текст").duration == 8.06


def test_transcript_text():
    segments = (Segment(0.0, 1.0, "Алло.", 0), Segment(0.5, 2.0, "Здравствуйте!", 1))
    transcript = Transcript(segments, duration=2.0, channels=2, model="m")
    assert transcript.text == "Алло. Здравствуйте!"
    assert transcript.source is None


def test_empty_transcript():
    assert Transcript((), duration=0.0, channels=1, model="m").text == ""


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Segment(0.0, 1.0, "Да.").text = "Нет."
