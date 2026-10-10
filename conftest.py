import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session", autouse=True)
def doctest_files(doctest_namespace, tmp_path_factory):
    """Стереозапись для примеров из докстрингов: две секунды, 8 кГц"""
    path = tmp_path_factory.mktemp("doctest") / "call.wav"
    time = np.arange(16000) / 8000
    samples = 0.1 * np.stack([np.sin(2 * np.pi * 440 * time), np.sin(2 * np.pi * 220 * time)], 1)
    sf.write(path, samples, 8000, subtype="PCM_16")
    doctest_namespace["call_wav"] = path
