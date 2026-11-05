import os
import sys
import threading
from pathlib import Path
from typing import ClassVar

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

import stt_ru
from stt_ru import Segment, Transcriber, pipeline
from stt_ru.audio import Audio
from stt_ru.diarization import Turn
from stt_ru.exceptions import ParameterError
from stt_ru.pipeline import cut_points, same_signal
from stt_ru.vad import VadOptions

CALL = Path(__file__).parent / "data" / "call.wav"


class FakeRecognizer:
    instances: ClassVar[list["FakeRecognizer"]] = []
    fragments: ClassVar[list] = []

    def __init__(self, model="fake", **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.calls = []
        FakeRecognizer.instances.append(self)

    def recognize(self, waveforms, sample_rate):
        self.calls.append(([w.copy() for w in waveforms], sample_rate))
        found = [[(1.0, 2.0, "Второй."), (0.2, 0.8, "Первый."), (3.0, 3.5, "")]]
        found += [[(0.5, 1.5, "Ответ.")] for _ in waveforms[1:]]
        return found

    def recognize_words(self, waveforms, sample_rate):
        assert len(waveforms) == 1
        return [FakeRecognizer.fragments]


@pytest.fixture
def fake(monkeypatch):
    FakeRecognizer.instances.clear()
    monkeypatch.setattr(pipeline, "Recognizer", FakeRecognizer)
    return FakeRecognizer.instances


class FakeDiarizer:
    instances: ClassVar[list["FakeDiarizer"]] = []
    turns: ClassVar[list[Turn]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeDiarizer.instances.append(self)

    def diarize(self, samples, sample_rate, speakers=None):
        self.calls.append((samples.copy(), sample_rate, speakers))
        return FakeDiarizer.turns


@pytest.fixture
def diarizing(fake, monkeypatch):
    FakeDiarizer.instances.clear()
    monkeypatch.setattr(pipeline, "Diarizer", FakeDiarizer)
    monkeypatch.setattr(FakeDiarizer, "turns", [Turn(0.0, 1.0, 2), Turn(1.0, 2.0, 1)])
    words = [(0.2, 0.6, "Алло."), (1.2, 1.5, "Да,"), (1.5, 1.8, "слушаю.")]
    monkeypatch.setattr(FakeRecognizer, "fragments", [(0.2, 1.8, "Алло. Да, слушаю.", words)])
    return FakeDiarizer.instances


def mono(rate=8000, seconds=2.0):
    return Audio(np.ones((int(rate * seconds), 1), dtype=np.float32) * 0.1, rate)


def stereo(rate=8000, seconds=1.0):
    count = int(rate * seconds)
    samples = np.random.default_rng(0).normal(0, 0.05, (count, 2)).astype(np.float32)
    samples[0] = (0.25, 0.75)
    return samples


def test_arguments(fake, tmp_path):
    vad = VadOptions(threshold=0.4)
    Transcriber("t-tech/t-one", quantization=None, model_dir=tmp_path, providers=["CPU"], vad=vad)
    (recognizer,) = fake
    assert recognizer.model == "t-tech/t-one"
    assert recognizer.kwargs == {
        "quantization": None,
        "model_dir": tmp_path,
        "providers": ["CPU"],
        "vad": vad,
    }


def test_channels(fake):
    transcript = Transcriber().transcribe_audio(Audio(stereo(), 8000), source="call.wav")
    ((waveforms, rate),) = fake[0].calls
    assert rate == 8000
    assert [float(w[0]) for w in waveforms] == [0.25, 0.75]
    assert transcript.segments == (
        Segment(0.2, 0.8, "Первый.", 0),
        Segment(0.5, 1.5, "Ответ.", 1),
        Segment(1.0, 2.0, "Второй.", 0),
    )
    assert (transcript.channels, transcript.duration) == (2, 1.0)
    assert (transcript.model, transcript.source) == ("gigaam-v3-e2e-rnnt", "call.wav")
    assert transcript.channel_map == (0, 1)


def test_same_start_ordered_by_channel(monkeypatch, fake):
    transcriber = Transcriber()
    monkeypatch.setattr(
        transcriber.recognizer, "recognize", lambda w, r: [[(1.0, 2.0, "Б")], [(1.0, 1.5, "А")]]
    )
    segments = transcriber.transcribe_audio(Audio(stereo(), 8000)).segments
    assert [s.channel for s in segments] == [0, 1]


def test_mixed_channels(fake):
    transcript = Transcriber(split_channels=False).transcribe_audio(Audio(stereo(), 8000))
    ((waveforms, _),) = fake[0].calls
    assert len(waveforms) == 1
    assert float(waveforms[0][0]) == 0.5
    assert {segment.channel for segment in transcript.segments} == {None}
    assert transcript.channels == 1
    assert transcript.channel_map == (None, None)


def test_mono(fake):
    transcript = Transcriber().transcribe_audio(Audio(stereo()[:, :1], 8000))
    assert transcript.channels == 1
    assert transcript.segments[0] == Segment(0.2, 0.8, "Первый.")


def test_identical_channels(fake):
    samples = np.repeat(stereo()[:, :1], 2, axis=1)
    transcript = Transcriber().transcribe_audio(Audio(samples, 8000))
    ((waveforms, _),) = fake[0].calls
    assert len(waveforms) == 1
    assert transcript.channels == 1
    assert {segment.channel for segment in transcript.segments} == {None}


def test_nearly_identical_channels(fake):
    rng = np.random.default_rng(0)
    left = rng.normal(0, 0.1, 8000).astype(np.float32)
    right = left + rng.normal(0, 0.001, 8000).astype(np.float32)
    transcript = Transcriber().transcribe_audio(Audio(np.stack([left, right], 1), 8000))
    assert transcript.channels == 1


def test_different_channels_kept(fake):
    rng = np.random.default_rng(0)
    left = rng.normal(0, 0.1, 8000).astype(np.float32)
    right = left + rng.normal(0, 0.01, 8000).astype(np.float32)
    transcript = Transcriber().transcribe_audio(Audio(np.stack([left, right], 1), 8000))
    assert transcript.channels == 2


def test_tracks_are_views(fake):
    samples = stereo()
    transcriber = Transcriber()
    tracks, _ = transcriber._tracks(Audio(samples, 8000))
    assert all(np.shares_memory(track, samples) for _, track in tracks)


def test_duplicate_channel_among_others(fake):
    samples = np.concatenate([stereo(), stereo()[:, :1]], axis=1)
    transcript = Transcriber().transcribe_audio(Audio(samples, 8000))
    ((waveforms, _),) = fake[0].calls
    assert [float(w[0]) for w in waveforms] == [0.25, 0.75]
    assert transcript.channels == 2
    assert {segment.channel for segment in transcript.segments} == {0, 1}
    assert transcript.channel_map == (0, 1, 0)


def test_quieter_and_shifted_copy_is_same_channel(fake):
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.1, 8100).astype(np.float32)
    # Второй канал на 6 дБ тише и позже на 2 мс: стереозапись одного источника
    samples = np.stack([signal[16:], 0.5 * signal[:-16]], axis=1)
    transcript = Transcriber().transcribe_audio(Audio(samples, 8000))
    assert transcript.channels == 1
    assert transcript.channel_map == (None, None)


class TestSameSignal:
    def test_silent_channels(self):
        silence = np.zeros(100, dtype=np.float32)
        signal = np.ones(100, dtype=np.float32)
        assert same_signal(silence, silence)
        assert not same_signal(silence, signal)
        assert not same_signal(signal, silence)

    def test_empty(self):
        empty = np.zeros(0, dtype=np.float32)
        assert same_signal(empty, empty, max_lag=80)

    def test_lag_found_in_loud_part(self, monkeypatch):
        monkeypatch.setattr(pipeline, "LAG_FRAME", 256)
        rng = np.random.default_rng(1)
        signal = np.zeros(4096, dtype=np.float32)
        signal[3000:3800] = rng.normal(0, 0.3, 800)
        assert same_signal(signal[5:], signal[:-5], max_lag=8)
        assert same_signal(signal[:-5], signal[5:], max_lag=8)
        assert not same_signal(signal[20:], signal[:-20], max_lag=8)


def test_empty_audio(fake):
    transcript = Transcriber().transcribe_audio(Audio(np.zeros((0, 2), dtype=np.float32), 8000))
    assert fake[0].calls == []
    assert transcript.segments == ()


def test_windows(fake, monkeypatch):
    monkeypatch.setattr(pipeline, "WINDOW_SECONDS", 1.0)
    monkeypatch.setattr(pipeline, "CUT_SEARCH_SECONDS", 0.25)
    rate = 8000
    samples = stereo(rate=rate, seconds=2.2)
    samples[int(1.0 * rate) : int(1.4 * rate)] = 0
    transcript = Transcriber().transcribe_audio(Audio(samples, rate))
    lengths = [[len(w) for w in waveforms] for waveforms, _ in fake[0].calls]
    cut = lengths[0][0]
    assert int(1.0 * rate) <= cut < int(1.4 * rate)
    assert lengths == [[cut, cut], [len(samples) - cut, len(samples) - cut]]
    first = [s for s in transcript.segments if s.text == "Первый."]
    offset = cut / rate
    assert [(s.start, s.end) for s in first] == [
        (0.2, 0.8),
        (round(offset + 0.2, 3), round(offset + 0.8, 3)),
    ]
    assert transcript.duration == 2.2


class TestCutPoints:
    @pytest.fixture(autouse=True)
    def short_windows(self, monkeypatch):
        monkeypatch.setattr(pipeline, "WINDOW_SECONDS", 1.0)
        monkeypatch.setattr(pipeline, "CUT_SEARCH_SECONDS", 0.5)

    def test_short(self):
        assert cut_points([np.ones(12000, dtype=np.float32)], 8000) == [0, 12000]

    def test_pauses(self):
        track = np.ones(8000 * 4, dtype=np.float32)
        track[7000:10200] = 0
        track[15000:18200] = 0
        cuts = cut_points([track], 8000)
        assert cuts[0] == 0 and cuts[-1] == len(track)
        assert 7000 <= cuts[1] < 10200
        assert 15000 <= cuts[2] < 18200

    def test_pause_beats_short_gap_inside_phrase(self):
        track = np.ones(8000 * 2, dtype=np.float32)
        track[5000:5320] = 0
        track[9000:11400] = 0.05
        assert 9000 <= cut_points([track], 8000)[1] < 11400

    def test_noisy_channel_does_not_outweigh(self):
        speech = np.ones(8000 * 2, dtype=np.float32)
        speech[9000:11400] = 0
        noise = np.full(8000 * 2, 3, dtype=np.float32)
        noise[5000:5320] = 0
        assert 9000 <= cut_points([speech, noise], 8000)[1] < 11400

    def test_digital_silence_channel_does_not_outweigh(self):
        speech = np.ones(8000 * 2, dtype=np.float32)
        speech[9000:11400] = 0
        clicks = np.zeros(8000 * 2, dtype=np.float32)
        for start in range(0, len(clicks), 800):
            if not 5000 <= start < 7400:
                clicks[start : start + 160] = 0.001
        assert 9000 <= cut_points([speech, clicks], 8000)[1] < 11400

    def test_last_window_absorbs_tail(self):
        assert cut_points([np.ones(8000 + 4000, dtype=np.float32)], 8000) == [0, 12000]
        assert len(cut_points([np.ones(8000 + 4001, dtype=np.float32)], 8000)) == 3


def test_transcribe_file(fake, tmp_path):
    path = tmp_path / "звонок.wav"
    sf.write(path, stereo(), 8000, subtype="PCM_16")
    transcript = Transcriber().transcribe(path)
    assert transcript.source == "звонок.wav"
    assert fake[0].calls[0][1] == 8000


def test_resample_unsupported_rate_once(fake, tmp_path, monkeypatch):
    path = tmp_path / "call.wav"
    sf.write(path, stereo(rate=12000), 12000, subtype="PCM_16")
    loads = []
    load = pipeline.load_audio

    def spy(*args, **kwargs):
        loads.append(kwargs["sample_rate"])
        return load(*args, **kwargs)

    monkeypatch.setattr(pipeline, "load_audio", spy)
    transcript = Transcriber(ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()).transcribe(path)
    assert loads == [16000]
    assert fake[0].calls[0][1] == 16000
    assert transcript.duration == pytest.approx(1.0, abs=0.01)


def test_ffmpeg_format_decoded_once_at_16_khz(fake, tmp_path, monkeypatch):
    path = tmp_path / "call.wav"
    sf.write(path, stereo(rate=8000), 8000, subtype="PCM_16")
    monkeypatch.setattr(
        pipeline.sf, "info", lambda path: (_ for _ in ()).throw(sf.LibsndfileError(1))
    )
    loads = []
    load = pipeline.load_audio

    def spy(*args, **kwargs):
        loads.append(kwargs["sample_rate"])
        return load(*args, **kwargs)

    monkeypatch.setattr(pipeline, "load_audio", spy)
    transcript = Transcriber(ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()).transcribe(path)
    assert loads == [16000]
    assert fake[0].calls[0][1] == 16000
    assert transcript.duration == pytest.approx(1.0, abs=0.01)


def test_load(fake, tmp_path):
    path = tmp_path / "call.wav"
    sf.write(path, stereo(rate=12000), 12000, subtype="PCM_16")
    transcriber = Transcriber(ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
    audio = transcriber.load(path)
    assert (audio.sample_rate, audio.num_channels) == (16000, 2)
    with pytest.raises(stt_ru.exceptions.AudioError, match=r"Запись длиннее 0\.5 с"):
        transcriber.load(path, max_duration=0.5)
    assert fake[0].calls == []


def test_missing_file(fake, tmp_path):
    with pytest.raises(FileNotFoundError):
        Transcriber().transcribe(tmp_path / "missing.wav")


@pytest.mark.skipif(sys.platform == "win32", reason="именованных каналов mkfifo в Windows нет")
def test_named_pipe_is_not_read(fake, tmp_path):
    pipe = tmp_path / "call.wav"
    os.mkfifo(pipe)
    with pytest.raises(FileNotFoundError, match="Файл не найден"):
        Transcriber().load(pipe)


def test_package_exports():
    assert stt_ru.Transcriber is Transcriber


class TestDiarization:
    def test_fragment_split_by_words(self, diarizing):
        transcript = Transcriber(diarize=True).transcribe_audio(mono())
        assert transcript.segments == (
            Segment(0.2, 1.2, "Алло.", speaker_id=1),
            Segment(1.2, 1.8, "Да, слушаю.", speaker_id=2),
        )
        assert transcript.channels == 1
        assert transcript.channel_map == (None,)
        (samples, rate, speakers) = diarizing[0].calls[0]
        assert (len(samples), rate, speakers) == (16000, 8000, None)

    def test_not_loaded_by_default(self, diarizing):
        assert Transcriber().diarizer is None
        assert diarizing == []

    def test_providers_passed(self, diarizing):
        Transcriber(diarize=True, providers=["CPU"])
        assert diarizing[0].kwargs == {"providers": ["CPU"]}

    def test_speakers_turn_on_diarization(self, diarizing):
        Transcriber(speakers=2).transcribe_audio(mono())
        assert diarizing[0].calls[0][2] == 2

    @pytest.mark.parametrize("speakers", [0, -2])
    def test_wrong_speakers(self, diarizing, speakers):
        with pytest.raises(ParameterError, match="Число говорящих"):
            Transcriber(speakers=speakers)

    def test_without_word_times(self, diarizing, monkeypatch):
        monkeypatch.setattr(FakeRecognizer, "fragments", [(0.2, 1.8, "Целиком.", None)])
        monkeypatch.setattr(FakeDiarizer, "turns", [Turn(0.0, 0.5, 4), Turn(0.5, 2.0, 7)])
        (segment,) = Transcriber(diarize=True).transcribe_audio(mono()).segments
        assert segment == Segment(0.2, 1.8, "Целиком.", speaker_id=1)

    def test_no_speech_found(self, diarizing, monkeypatch):
        monkeypatch.setattr(FakeDiarizer, "turns", [])
        segments = Transcriber(diarize=True).transcribe_audio(mono()).segments
        assert segments == (Segment(0.2, 1.8, "Алло. Да, слушаю.", speaker_id=1),)

    def test_empty_text_dropped(self, diarizing, monkeypatch):
        monkeypatch.setattr(FakeRecognizer, "fragments", [(0.2, 0.3, "", [])])
        assert Transcriber(diarize=True).transcribe_audio(mono()).segments == ()

    def test_numbered_without_gaps(self, diarizing, monkeypatch):
        monkeypatch.setattr(FakeDiarizer, "turns", [Turn(0.0, 2.0, 5)])
        (segment,) = Transcriber(diarize=True).transcribe_audio(mono()).segments
        assert segment.speaker_id == 1

    def test_different_channels_kept(self, diarizing):
        audio = Audio(stereo(), 8000)
        with pytest.warns(UserWarning, match="^звонок.wav: каналы разные"):
            transcript = Transcriber(diarize=True).transcribe_audio(audio, "звонок.wav")
        assert {segment.channel for segment in transcript.segments} == {0, 1}
        assert {segment.speaker_id for segment in transcript.segments} == {None}
        assert diarizing[0].calls == []

    def test_different_channels_without_name(self, diarizing):
        with pytest.warns(UserWarning, match="^Запись: каналы разные"):
            Transcriber(diarize=True).transcribe_audio(Audio(stereo(), 8000))

    def test_mixed_channels_diarized(self, diarizing):
        audio = Audio(stereo(), 8000)
        Transcriber(diarize=True, split_channels=False).transcribe_audio(audio)
        assert len(diarizing[0].calls) == 1

    def test_turned_on_for_one_record(self, diarizing):
        transcriber = Transcriber()
        assert diarizing == []
        first = transcriber.transcribe_audio(mono(), diarize=True)
        second = transcriber.transcribe_audio(mono(), speakers=2)
        plain = transcriber.transcribe_audio(mono())
        assert first.segments[0].speaker_id == 1
        assert second.segments[0].speaker_id == 1
        assert plain.segments[0].speaker_id is None
        assert len(diarizing) == 1
        assert [call[2] for call in diarizing[0].calls] == [None, 2]

    def test_turned_off_for_one_record(self, diarizing):
        transcriber = Transcriber(speakers=2)
        assert transcriber.transcribe_audio(mono(), diarize=False).segments[0].speaker_id is None
        assert diarizing[0].calls == []

    def test_default_speakers_kept(self, diarizing):
        Transcriber(speakers=2).transcribe_audio(mono(), diarize=True)
        assert diarizing[0].calls[0][2] == 2

    def test_file_options(self, diarizing, tmp_path):
        path = tmp_path / "rec.wav"
        sf.write(path, mono().samples, 8000)
        segments = Transcriber().transcribe(path, speakers=3).segments
        assert segments[0].speaker_id == 1
        assert diarizing[0].calls[0][2] == 3

    def test_loaded_once_from_threads(self, diarizing):
        transcriber = Transcriber()
        threads = [
            threading.Thread(
                target=transcriber.transcribe_audio, args=(mono(),), kwargs={"diarize": True}
            )
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(diarizing) == 1
        assert len(diarizing[0].calls) == 8

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"speakers": 0}, "Число говорящих должно быть положительным"),
            ({"diarize": False, "speakers": 2}, "разделение говорящих выключено"),
        ],
    )
    def test_wrong_options(self, diarizing, options, message):
        with pytest.raises(ParameterError, match=message):
            Transcriber().transcribe_audio(mono(), **options)

    def test_speaker_options(self, diarizing):
        assert Transcriber().speaker_options() == (False, None)
        assert Transcriber().speaker_options(diarize=True) == (True, None)
        assert Transcriber(diarize=True).speaker_options() == (True, None)
        assert Transcriber(speakers=4).speaker_options() == (True, 4)
        assert Transcriber(speakers=4).speaker_options(speakers=2) == (True, 2)
        assert Transcriber(speakers=4).speaker_options(diarize=False) == (False, None)

    def test_windows_keep_offset(self, diarizing, monkeypatch):
        monkeypatch.setattr(pipeline, "WINDOW_SECONDS", 1.0)
        monkeypatch.setattr(pipeline, "CUT_SEARCH_SECONDS", 0.1)
        words = [(0.2, 0.4, "Да.")]
        monkeypatch.setattr(FakeRecognizer, "fragments", [(0.2, 0.4, "Да.", words)])
        segments = Transcriber(diarize=True).transcribe_audio(mono(seconds=3.0)).segments
        assert len(segments) == 3
        assert [segment.start for segment in segments] == sorted(
            segment.start for segment in segments
        )
        assert segments[-1].start > 2.0


@pytest.mark.network
def test_real_diarization():
    transcriber = Transcriber(diarize=True, split_channels=False)
    segments = transcriber.transcribe(CALL).segments
    operator = next(s for s in segments if "благодарю" in s.text.lower())
    client = next(s for s in segments if s.text.lower().startswith("до свидания") and s.start > 12)
    assert operator.speaker == "Говорящий 1"
    assert client.speaker == "Говорящий 2"
    assert {s.speaker_id for s in segments} == {1, 2}
    exact = transcriber.transcribe(CALL, speakers=2).segments
    assert [s.speaker_id for s in exact] == [s.speaker_id for s in segments]


@pytest.mark.network
def test_real_model():
    transcript = Transcriber().transcribe(CALL)
    assert transcript.channels == 2
    assert transcript.model == "gigaam-v3-e2e-rnnt"
    operator = " ".join(s.text for s in transcript.segments if s.channel == 1).lower()
    client = " ".join(s.text for s in transcript.segments if s.channel == 0).lower()
    assert "благодарю вас за ожидание" in operator
    assert "до свидания" in operator
    assert "," in operator
    assert "до свидания" in client
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert all(0 <= s.start < s.end <= transcript.duration for s in transcript.segments)


@pytest.mark.network
def test_real_model_windows(monkeypatch):
    monkeypatch.setattr(pipeline, "WINDOW_SECONDS", 9.0)
    monkeypatch.setattr(pipeline, "CUT_SEARCH_SECONDS", 1.0)
    transcript = Transcriber().transcribe(CALL)
    operator = " ".join(s.text for s in transcript.segments if s.channel == 1).lower()
    assert "благодарю вас за ожидание" in operator
    assert "до свидания" in operator
