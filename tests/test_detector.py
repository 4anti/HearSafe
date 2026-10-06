from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hearsafe import AlertRule, Detector
from hearsafe.audio import log_mel, mono_audio
from hearsafe.cli import analyse_file
from hearsafe.types import Label


class FakeModel:
    def __init__(self, window=16000, hop=8000):
        self.manifest = SimpleNamespace(
            model_id="test",
            sample_rate=16000,
            window_samples=window,
            hop_samples=hop,
            labels=[Label("knock", "Knock"), Label("noise", "Noise")],
        )
        self.windows = []

    def predict(self, window):
        self.windows.append(window.copy())
        return np.array([0.9, 0.2], dtype=np.float32)


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_chunking_and_resampling_parity(rate):
    audio = (np.sin(np.arange(rate * 2 + 123) * 0.0123) * 0.1).astype(np.float32)
    full, chunks = FakeModel(), FakeModel()
    one, many = Detector(full), Detector(chunks)
    expected = one.process(audio, rate) + one.flush()
    actual = []
    offset = 0
    for size in [1, 77, 4000, 293, 9000] * 50:
        if offset >= len(audio):
            break
        actual.extend(many.process(audio[offset : offset + size], rate))
        offset += size
    actual.extend(many.flush())
    assert [(f.start_ms, f.end_ms) for f in actual] == [(f.start_ms, f.end_ms) for f in expected]
    np.testing.assert_allclose(chunks.windows, full.windows, atol=1e-6)


def test_exact_window_flush_does_not_duplicate():
    model = FakeModel()
    detector = Detector(model)
    assert len(detector.process(np.zeros(16000, np.float32), 16000)) == 1
    assert detector.flush() == []
    assert detector.flush() == []
    with pytest.raises(ValueError, match="reset"):
        detector.process(np.zeros(1, np.float32), 16000)


def test_short_file_and_adapter_parity(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.full((4000, 2), 0.1, np.float32), 16000, subtype="FLOAT")
    detector = Detector(FakeModel())
    results = list(analyse_file(detector, path))
    assert len(results) == 1 and results[0].end_ms == 250
    assert results[0].top()[0].label == "Knock"
    assert results[0].events == []
    assert detector.model.windows[0][4000:].max() == 0


def test_invalid_input_and_rate_change():
    detector = Detector(FakeModel())
    for audio in [
        np.array([np.nan], np.float32),
        np.array([2.0], np.float32),
        np.array([1], np.int16),
    ]:
        with pytest.raises(ValueError):
            detector.process(audio, 16000)
    assert detector.process(np.array([], np.float32), 16000) == []
    detector.process(np.zeros(1, np.float32), 16000)
    with pytest.raises(ValueError, match="changed"):
        detector.process(np.zeros(1, np.float32), 48000)
    detector.reset()
    detector.process(np.zeros(1, np.float32), 48000)


def test_invalid_model_output_and_watchlist():
    model = FakeModel()
    detector = Detector(model)
    with pytest.raises(ValueError, match="Watchlist"):
        detector.set_alerts({"unknown": AlertRule()})
    model.predict = lambda audio: np.array([np.nan, 0])
    with pytest.raises(ValueError, match="invalid scores"):
        detector.process(np.zeros(16000, np.float32), 16000)
    with pytest.raises(ValueError, match="Watchlist"):
        Detector(FakeModel(), {"unknown": AlertRule()})
    model = FakeModel()
    model.predict = lambda audio: np.array([1.1, -0.1])
    with pytest.raises(ValueError, match="invalid scores"):
        Detector(model).process(np.zeros(16000, np.float32), 16000)


def test_stereo_and_frontend():
    np.testing.assert_allclose(mono_audio(np.array([[0.2, 0.4]], np.float32)), [0.3])
    features = log_mel(np.zeros(80000, np.float32))
    assert features.shape == (64, 498) and features.dtype == np.float32
    assert np.isfinite(features).all()
    with pytest.raises(ValueError):
        log_mel(np.zeros(79999, np.float32))
