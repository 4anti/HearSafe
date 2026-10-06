from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from hearsafe.microphone import MicrophoneError, MicrophoneSession, devices


def eventually(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    assert condition(), "worker did not reach the expected state"


class FakeDetector:
    def __init__(self):
        self.resets = 0
        self.blocks = []
        self.rules = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False

    def reset(self):
        self.resets += 1

    def set_alerts(self, rules):
        self.rules.append(rules)

    def process(self, audio, rate):
        self.entered.set()
        if self.block:
            assert self.release.wait(2)
        if self.fail:
            raise ValueError("invalid model result")
        self.blocks.append((audio, rate))
        return [{"frame": len(self.blocks)}]


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        self.kwargs["finished_callback"]()

    def close(self):
        self.closed = True

    def send(self, samples, status=False):
        self.kwargs["callback"](samples, len(samples), None, status)


class FakeSounddevice:
    def __init__(self):
        self.stream = None
        self.fail_open = False
        self.info = [
            {"name": "Output only", "max_input_channels": 0, "default_samplerate": 48000.0},
            {"name": "Headset", "max_input_channels": 2, "default_samplerate": 44100.0},
        ]

    def query_devices(self, device=None, kind=None):
        if kind == "input":
            assert device in (None, 1)
            return self.info[1]
        return self.info

    def check_input_settings(self, **kwargs):
        assert kwargs["samplerate"] == 44100
        if self.fail_open:
            raise RuntimeError("permission denied")

    def InputStream(self, **kwargs):
        self.stream = FakeStream(**kwargs)
        return self.stream


@pytest.fixture
def sd():
    fake = FakeSounddevice()
    with patch("hearsafe.microphone._sounddevice", return_value=fake):
        yield fake


def test_inventory_keeps_input_device_ids(sd):
    assert devices() == [{"id": 1, "name": "Headset", "channels": 2, "default_samplerate": 44100.0}]


def test_capture_native_rate_and_copy_callback_buffer(sd):
    detector = FakeDetector()
    frames, levels = [], []
    session = MicrophoneSession(detector, device=1, on_frame=frames.append, on_level=levels.append)
    try:
        session.start()
        assert session.running
        samples = np.full((500, 2), 0.25, dtype=np.float32)
        sd.stream.send(samples)
        samples.fill(0)
        eventually(lambda: len(frames) == 1)
        captured, rate = detector.blocks[0]
        assert rate == 44100
        np.testing.assert_allclose(captured, 0.25)
        assert levels == pytest.approx([0.25])
    finally:
        session.stop()
    assert sd.stream.closed
    assert not session.running
    assert session.last_error is None


def test_denied_device_reports_actionable_error_and_never_switches(sd):
    sd.fail_open = True
    session = MicrophoneSession(FakeDetector(), device=1)
    with pytest.raises(MicrophoneError, match="permission denied"):
        session.start()
    assert not session.running
    assert sd.stream is None
    assert "permission denied" in session.last_error


def test_disconnection_stops_capture_and_tells_user_to_reselect(sd):
    messages = []
    session = MicrophoneSession(FakeDetector(), device=1, on_status=messages.append)
    session.start()
    sd.stream.kwargs["finished_callback"]()
    eventually(lambda: sd.stream.closed)
    assert not session.running
    assert "disconnected" in session.last_error
    assert any("disconnected" in message and "Select" in message for message in messages)
    session.stop()


def test_hardware_gap_resets_detector_before_new_audio(sd):
    detector = FakeDetector()
    messages = []
    session = MicrophoneSession(detector, device=1, on_status=messages.append)
    try:
        session.start()
        reset_count = detector.resets
        sd.stream.send(np.zeros((20, 2), dtype=np.float32), status="input overflow")
        eventually(lambda: len(detector.blocks) == 1)
        assert detector.resets > reset_count
        assert session.dropped_chunks == 1
        eventually(lambda: any("Audio gap" in message for message in messages))
    finally:
        session.stop()


def test_queue_overflow_drops_backlog_and_resets_state(sd):
    detector = FakeDetector()
    detector.block = True
    messages = []
    session = MicrophoneSession(detector, device=1, queue_size=1, on_status=messages.append)
    try:
        session.start()
        sd.stream.send(np.zeros((20, 2), dtype=np.float32))
        assert detector.entered.wait(1)
        sd.stream.send(np.ones((20, 2), dtype=np.float32) * 0.1)
        sd.stream.send(np.ones((20, 2), dtype=np.float32) * 0.2)
        assert session.dropped_chunks == 1
        detector.release.set()
        eventually(lambda: len(detector.blocks) == 2)
        np.testing.assert_allclose(detector.blocks[-1][0], 0.2)
        assert detector.resets >= 2
        eventually(lambda: any("queue full" in message for message in messages))
    finally:
        detector.release.set()
        session.stop()


def test_inference_error_closes_device(sd):
    detector = FakeDetector()
    detector.fail = True
    messages = []
    session = MicrophoneSession(detector, device=1, on_status=messages.append)
    session.start()
    sd.stream.send(np.zeros((20, 2), dtype=np.float32))
    eventually(lambda: sd.stream.closed)
    assert not session.running
    assert any("invalid model result" in message for message in messages)
    assert "invalid model result" in session.last_error
    session.stop()


def test_watchlist_changes_run_on_inference_worker(sd):
    detector = FakeDetector()
    session = MicrophoneSession(detector, device=1)
    try:
        session.start()
        session.set_alerts({"knock": "rule"})
        eventually(lambda: detector.rules == [{"knock": "rule"}])
    finally:
        session.stop()


def test_no_optional_sounddevice_import_until_requested():
    with patch("hearsafe.microphone.importlib.import_module", side_effect=ImportError("missing")):
        session = MicrophoneSession(FakeDetector())
        with pytest.raises(MicrophoneError, match="runtime dependencies"):
            session.start()
        assert not session.running
