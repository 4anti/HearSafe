"""Behavior shared by the Python API, JSON command, and desktop file worker."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hearsafe import AlertRule, Detector
from hearsafe.audio import iter_wav_chunks
from hearsafe.cli import main as cli_main
from hearsafe.models import default_model_dir, load_model
from hearsafe.types import Label


class ConstantModel:
    def __init__(self):
        self.manifest = SimpleNamespace(
            model_id="test-gap",
            sample_rate=16000,
            window_samples=15600,
            hop_samples=7680,
            labels=[Label("knock", "Knock")],
        )

    def predict(self, window):
        return np.array([0.95], dtype=np.float32)


def test_gap_reset_discards_pending_audio_and_rearms_existing_watchlist():
    detector = Detector(ConstantModel())
    detector.set_alerts({"knock": AlertRule(transient=True)})
    assert detector.process(np.full(15000, 0.2, np.float32), 16000) == []
    detector.reset()
    # These samples must not complete the window that preceded the gap.
    assert detector.process(np.full(600, 0.2, np.float32), 16000) == []
    frame = detector.process(np.full(15000, 0.2, np.float32), 16000)[0]
    assert frame.start_ms == 0
    assert len(frame.events) == 1
    # A reset also clears the active alert and its cooldown, retaining policy.
    detector.reset()
    next_frame = detector.process(np.full(15600, 0.2, np.float32), 16000)[0]
    assert len(next_frame.events) == 1


@pytest.mark.parametrize("last_error, expected_exit", [(None, 0), ("Microphone disconnected", 1)])
def test_cli_listen_reports_unexpected_session_failure(
    monkeypatch, capsys, last_error, expected_exit
):
    stopped = []

    class FinishedSession:
        def __init__(self, *args, **kwargs):
            self.running = False
            self.last_error = last_error

        def start(self):
            pass

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr("hearsafe.cli.load_model", lambda path: ConstantModel())
    monkeypatch.setattr("hearsafe.microphone.MicrophoneSession", FinishedSession)
    assert cli_main(["listen", "--duration", "1", "--json"]) == expected_exit
    output = capsys.readouterr()
    assert output.out == ""
    assert stopped == [True]
    if last_error:
        assert last_error in output.err


def assert_same_analysis(expected, actual):
    assert actual["schema_version"] == expected.schema_version
    assert actual["model_id"] == expected.model_id
    assert actual["start_ms"] == expected.start_ms
    assert actual["end_ms"] == expected.end_ms
    candidates = expected.top(5)
    assert [item["label_id"] for item in actual["predictions"]] == [
        item.label_id for item in candidates
    ]
    assert [item["label"] for item in actual["predictions"]] == [item.label for item in candidates]
    np.testing.assert_allclose(
        [item["score"] for item in actual["predictions"]],
        [item.score for item in candidates],
        rtol=0,
        atol=1e-6,
    )
    assert actual["events"] == [event.to_dict() for event in expected.events]


def test_corrupt_wav_returns_error_on_diagnostic_stream(tmp_path, monkeypatch, capsys):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"this is not a WAV file")
    monkeypatch.setattr("hearsafe.cli.load_model", lambda model_path: ConstantModel())
    assert cli_main(["detect", str(path), "--json"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Cannot read audio file" in output.err


def test_real_yamnet_file_matches_python_json_cli_and_desktop_worker(tmp_path, monkeypatch, capsys):
    model_path = default_model_dir()
    if not (model_path / "model.tflite").is_file():
        pytest.skip("Download the pinned YAMNet model to run interface integration")
    pytest.importorskip("ai_edge_litert.interpreter")
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Desktop display is unavailable")
    root.withdraw()
    from hearsafe.gui import HearSafeApp

    # Synthetic, redistributable audio exercises stereo conversion, continuous
    # 44.1kHz resampling, overlapping windows, and the padded file tail.
    rate = 44100
    times = np.arange(round(rate * 2.3), dtype=np.float64) / rate
    rng = np.random.default_rng(42)
    left = (0.12 * np.sin(2 * np.pi * 440 * times) + rng.normal(0, 0.02, len(times))).astype(
        np.float32
    )
    right = (0.1 * np.sin(2 * np.pi * 880 * times)).astype(np.float32)
    audio_path = tmp_path / "stereo fixture.wav"
    sf.write(audio_path, np.column_stack((left, right)), rate, subtype="PCM_16")
    monkeypatch.setenv("HEARSAFE_MODEL_DIR", str(model_path.resolve()))
    # Inventory is unnecessary for a file test; never access a real microphone.
    monkeypatch.setattr("hearsafe.gui.devices", lambda: [])
    app = HearSafeApp(root)
    try:
        detector = Detector(load_model(model_path))
        expected = []
        for chunk, native_rate in iter_wav_chunks(audio_path):
            expected.extend(detector.process(chunk, native_rate))
        expected.extend(detector.flush())
        assert len(expected) >= 3
        assert expected[-1].end_ms == pytest.approx(2300, abs=0.1)
        capsys.readouterr()
        assert cli_main(["detect", str(audio_path), "--model", str(model_path), "--json"]) == 0
        cli_records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert len(cli_records) == len(expected)
        for frame, cli_record in zip(expected, cli_records, strict=True):
            assert_same_analysis(frame, cli_record)

        # Exercise asynchronous model load and the real desktop file worker.
        app._load_selected(str(audio_path))
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            root.update()
            if app._records and not app._busy:
                break
            time.sleep(0.005)
        assert not app._busy, app._status.get()
        assert len(app._labels) == 521
        assert len(app._records) == len(expected), app._status.get()
        for frame, record in zip(expected, app._records, strict=True):
            assert_same_analysis(frame, record["analysis"])
        metrics = app._metric_reports()[0]
        assert metrics["windows"] == len(expected)
        assert metrics["inference_p95_ms"] > 0
        assert metrics["sampled_peak_rss_bytes"] > 0

        # A queued result from the previous model must never repopulate the UI.
        previous_generation = app._generation
        label_id = next(iter(app._labels))
        app._rules[label_id] = AlertRule()
        app._alerts_enabled.set(True)
        app._switch_model()
        app._post("frame", expected[0], previous_generation)
        app._poll()
        assert app._model is None
        assert app._last_frames == []
        assert app._labels == {} and app._rules == {}
        assert not app._alerts_enabled.get()
        assert app._candidate_tree.get_children() == ()
    finally:
        app.close()
