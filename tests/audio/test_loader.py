import builtins
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

from stt_ru.audio import Audio, find_ffmpeg, load_audio, loader
from stt_ru.exceptions import AudioError, SttRuError

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def stereo(rate=8000, seconds=1.0):
    time = np.arange(int(rate * seconds)) / rate
    left = 0.25 * np.sin(2 * np.pi * 440 * time)
    right = 0.5 * np.sin(2 * np.pi * 220 * time)
    return np.stack([left, right], axis=1).astype(np.float32)


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "call.wav"
    sf.write(path, stereo(), 8000, subtype="PCM_16")
    return path


@pytest.fixture
def m4a(tmp_path):
    path = tmp_path / "call.m4a"
    command = [
        FFMPEG,
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
    ]
    command += ["-ac", "2", "-ar", "16000", "-c:a", "aac", str(path)]
    subprocess.run(command, check=True)
    return path


@pytest.fixture
def pipe_flac(tmp_path):
    path = tmp_path / "pipe.flac"
    command = [
        FFMPEG,
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
    ]
    command += ["-ar", "16000", "-f", "flac", "pipe:1"]
    path.write_bytes(subprocess.run(command, capture_output=True, check=True).stdout)
    return path


@pytest.fixture
def zero_sizes_wav(tmp_path):
    path = tmp_path / "zero.wav"
    sf.write(path, stereo(), 8000, subtype="PCM_16")
    data = bytearray(path.read_bytes())
    data[4:8] = bytes(4)
    chunk = data.index(b"data")
    data[chunk + 4 : chunk + 8] = bytes(4)
    path.write_bytes(bytes(data))
    return path


@pytest.fixture
def no_ffmpeg(monkeypatch):
    monkeypatch.setattr(loader, "find_ffmpeg", lambda ffmpeg=None: None)


class TestAudio:
    def test_properties(self):
        audio = Audio(stereo(), 8000)
        assert audio.num_channels == 2
        assert audio.duration == 1.0

    def test_empty(self):
        audio = Audio(np.zeros((0, 1), dtype=np.float32), 16000)
        assert audio.duration == 0.0
        assert audio.mono().shape == (0,)

    @pytest.mark.parametrize(
        "samples",
        [
            np.zeros(10, dtype=np.float32),
            np.zeros((10, 2), dtype=np.float64),
            np.zeros((10, 2, 1), dtype=np.float32),
        ],
        ids=["1d", "float64", "3d"],
    )
    def test_wrong_samples(self, samples):
        with pytest.raises(AudioError, match="float32"):
            Audio(samples, 8000)

    def test_no_channels(self):
        with pytest.raises(AudioError, match="нет каналов"):
            Audio(np.zeros((10, 0), dtype=np.float32), 8000)

    @pytest.mark.parametrize("rate", [0, -8000])
    def test_wrong_sample_rate(self, rate):
        with pytest.raises(AudioError, match="положительной"):
            Audio(stereo(), rate)

    def test_channel(self):
        samples = stereo()
        right = Audio(samples, 8000).channel(1)
        assert right.ndim == 1
        assert right.flags.c_contiguous
        assert np.array_equal(right, samples[:, 1])

    @pytest.mark.parametrize("index", [-1, 2])
    def test_missing_channel(self, index):
        with pytest.raises(AudioError, match=f"Канала {index} нет: в записи каналов 2"):
            Audio(stereo(), 8000).channel(index)

    def test_mono(self):
        samples = stereo()
        mixed = Audio(samples, 8000).mono()
        assert mixed.dtype == np.float32
        assert np.allclose(mixed, samples.mean(axis=1))

    def test_mono_of_mono(self):
        samples = stereo()[:, :1]
        assert np.array_equal(Audio(samples, 8000).mono(), samples[:, 0])

    def test_error_hierarchy(self):
        assert issubclass(AudioError, SttRuError)


class TestLoadAudio:
    def test_wav(self, wav):
        audio = load_audio(wav)
        assert (audio.num_channels, audio.sample_rate) == (2, 8000)
        assert audio.samples.dtype == np.float32
        assert np.allclose(audio.samples, stereo(), atol=1e-4)

    def test_str_path(self, wav):
        assert load_audio(str(wav)).num_channels == 2

    @pytest.mark.parametrize(("suffix", "subtype"), [("flac", "PCM_16"), ("ogg", "VORBIS")])
    def test_soundfile_formats(self, tmp_path, no_ffmpeg, suffix, subtype):
        path = tmp_path / f"call.{suffix}"
        sf.write(path, stereo(), 8000, subtype=subtype)
        audio = load_audio(path)
        assert (audio.num_channels, audio.sample_rate, audio.duration) == (2, 8000, 1.0)

    @pytest.mark.skipif("MP3" not in sf.available_formats(), reason="libsndfile без MP3")
    def test_mp3(self, tmp_path, no_ffmpeg):
        path = tmp_path / "call.mp3"
        sf.write(path, stereo(rate=16000), 16000, format="MP3")
        audio = load_audio(path)
        assert (audio.num_channels, audio.sample_rate) == (2, 16000)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            load_audio(tmp_path / "missing.wav")

    def test_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            load_audio(tmp_path)

    def test_ffmpeg_format(self, m4a):
        audio = load_audio(m4a, ffmpeg=FFMPEG)
        assert (audio.num_channels, audio.sample_rate) == (2, 16000)
        assert audio.duration == pytest.approx(1.0, abs=0.05)
        assert np.abs(audio.samples).max() > 0.05

    def test_ffmpeg_format_without_ffmpeg(self, m4a, no_ffmpeg):
        with pytest.raises(AudioError, match=r"формат не читается .* ffmpeg не найден"):
            load_audio(m4a)

    def test_resample(self, wav):
        audio = load_audio(wav, sample_rate=16000, ffmpeg=FFMPEG)
        assert (audio.num_channels, audio.sample_rate) == (2, 16000)
        assert audio.duration == pytest.approx(1.0, abs=0.01)

    def test_resample_ffmpeg_format(self, m4a):
        audio = load_audio(m4a, sample_rate=8000, ffmpeg=FFMPEG)
        assert audio.sample_rate == 8000

    def test_resample_without_ffmpeg(self, wav, no_ffmpeg):
        with pytest.raises(AudioError, match="смена частоты дискретизации с 8000 на 16000 Гц"):
            load_audio(wav, sample_rate=16000)

    def test_same_rate_skips_ffmpeg(self, wav, no_ffmpeg):
        assert load_audio(wav, sample_rate=8000).sample_rate == 8000

    def test_ffmpeg_failure(self, tmp_path):
        path = tmp_path / "broken.m4a"
        path.write_bytes(b"not an audio file")
        with pytest.raises(AudioError, match="ffmpeg не прочитал файл"):
            load_audio(path, ffmpeg=FFMPEG)

    def test_ffmpeg_not_runnable(self, m4a, tmp_path):
        with pytest.raises(AudioError, match="Не удалось запустить ffmpeg"):
            load_audio(m4a, ffmpeg=tmp_path / "ffmpeg")

    @pytest.mark.parametrize("rate", [0, -16000])
    def test_wrong_sample_rate(self, wav, rate):
        with pytest.raises(AudioError, match="положительной"):
            load_audio(wav, sample_rate=rate, ffmpeg=FFMPEG)

    def test_unknown_length(self, pipe_flac):
        assert sf.info(pipe_flac).frames == loader._UNKNOWN_FRAMES
        audio = load_audio(pipe_flac, ffmpeg=FFMPEG)
        assert (audio.num_channels, audio.sample_rate, audio.duration) == (1, 16000, 2.0)

    def test_unknown_length_without_ffmpeg(self, pipe_flac, no_ffmpeg):
        with pytest.raises(AudioError, match="длина записи не указана в заголовке"):
            load_audio(pipe_flac)

    @pytest.mark.skipif("MP3" not in sf.available_formats(), reason="libsndfile без MP3")
    def test_concatenated_mp3(self, tmp_path):
        parts = []
        for seconds in (1.0, 3.0):
            part = tmp_path / f"part{seconds}.mp3"
            sf.write(part, stereo(rate=16000, seconds=seconds), 16000, format="MP3")
            parts.append(part.read_bytes())
        path = tmp_path / "concat.mp3"
        path.write_bytes(b"".join(parts))
        assert sf.info(path).duration < 2
        assert load_audio(path, ffmpeg=FFMPEG).duration == pytest.approx(4.0, abs=0.2)

    def test_zero_sizes_in_header(self, zero_sizes_wav):
        assert sf.info(zero_sizes_wav).frames == 0
        audio = load_audio(zero_sizes_wav, ffmpeg=FFMPEG)
        assert audio.duration == pytest.approx(1.0, abs=0.01)
        assert np.allclose(audio.samples, stereo(), atol=1e-4)

    def test_zero_sizes_without_ffmpeg(self, zero_sizes_wav, no_ffmpeg):
        assert load_audio(zero_sizes_wav).duration == 0.0

    def test_resample_skips_soundfile_read(self, wav, monkeypatch):
        read = sf.SoundFile.read

        def guarded(self, *args, **kwargs):
            assert self.name != str(wav), "soundfile не должен читать исходный файл"
            return read(self, *args, **kwargs)

        monkeypatch.setattr(sf.SoundFile, "read", guarded)
        assert load_audio(wav, sample_rate=16000, ffmpeg=FFMPEG).sample_rate == 16000

    def test_ffmpeg_writes_rf64(self, wav, monkeypatch):
        commands = []
        run = subprocess.run

        def record(command, **kwargs):
            commands.append(command)
            return run(command, **kwargs)

        monkeypatch.setattr(loader.subprocess, "run", record)
        load_audio(wav, sample_rate=16000, ffmpeg=FFMPEG)
        (command,) = commands
        assert command[command.index("-rf64") + 1] == "auto"

    def test_protocol_like_name(self, tmp_path):
        path = tmp_path / "concat:call.wav"
        sf.write(path, stereo(), 8000, subtype="PCM_16")
        assert load_audio(path, sample_rate=16000, ffmpeg=FFMPEG).sample_rate == 16000


class TestFindFfmpeg:
    def test_argument(self, tmp_path):
        assert find_ffmpeg(tmp_path / "ffmpeg") == str(tmp_path / "ffmpeg")

    def test_path(self, monkeypatch):
        monkeypatch.setattr(loader.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        assert find_ffmpeg() == "/usr/bin/ffmpeg"

    def test_imageio(self, monkeypatch):
        monkeypatch.setattr(loader.shutil, "which", lambda name: None)
        assert find_ffmpeg() == FFMPEG

    def test_imageio_missing(self, monkeypatch):
        monkeypatch.setattr(loader.shutil, "which", lambda name: None)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "imageio_ffmpeg":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "imageio_ffmpeg")
        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert find_ffmpeg() is None

    def test_imageio_without_binary(self, monkeypatch):
        monkeypatch.setattr(loader.shutil, "which", lambda name: None)

        def no_binary():
            raise RuntimeError("no ffmpeg binary")

        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", no_binary)
        assert find_ffmpeg() is None
