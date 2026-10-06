"""Optional microphone adapter for the shared detector.

Importing this module does not open a device. The capture callback only copies
audio to a bounded queue; model inference runs in a separate worker.
"""

from __future__ import annotations

import importlib
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any, Self

import numpy as np

_LOG = logging.getLogger(__name__)


class MicrophoneError(RuntimeError):
    """A microphone cannot be opened or continued."""


def _sounddevice() -> Any:
    try:
        return importlib.import_module("sounddevice")
    except (ImportError, OSError) as exc:
        raise MicrophoneError(
            "Microphone support is unavailable. Install the runtime dependencies "
            "or use the complete portable download."
        ) from exc


def devices() -> list[dict[str, Any]]:
    """List actual input devices, retaining sounddevice's numeric device IDs."""
    try:
        return [
            {
                "id": index,
                "name": str(info["name"]),
                "channels": int(info["max_input_channels"]),
                "default_samplerate": float(info["default_samplerate"]),
            }
            for index, info in enumerate(_sounddevice().query_devices())
            if int(info["max_input_channels"]) > 0
        ]
    except MicrophoneError:
        raise
    except Exception as exc:
        raise MicrophoneError(f"Cannot list microphones: {exc}") from exc


class MicrophoneSession:
    """Capture a selected microphone without storing raw audio.

    Callbacks run on worker threads, never the caller's UI thread. ``on_level``
    receives a linear RMS amplitude, ``on_status`` a readable diagnostic, and
    ``on_frame`` the detector's unchanged AnalysisFrame. ``start`` may raise a
    MicrophoneError; later errors stop the session and arrive through on_status.
    """

    def __init__(
        self,
        detector: Any,
        device: int | str | None = None,
        on_frame: Callable[[Any], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        *,
        queue_size: int = 16,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.detector = detector
        self.device = device
        self.on_frame = on_frame or (lambda frame: None)
        self.on_status = on_status or (lambda message: None)
        self.on_level = on_level or (lambda level: None)
        self.sample_rate: int | None = None
        self.dropped_chunks = 0
        self.last_error: str | None = None
        self._audio: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(queue_size)
        self._diagnostics: queue.Queue[str] = queue.Queue(32)
        self._controls: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._stream: Any = None
        self._worker: threading.Thread | None = None
        self._generation = 0
        self._running = False
        self._stream_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running and not self._stop.is_set()

    def set_alerts(self, rules: dict[str, Any]) -> None:
        """Apply watchlist edits on the inference worker, between audio chunks."""
        self._controls.put(dict(rules))

    def _status(self, message: str) -> None:
        try:
            self.on_status(message)
        except Exception:  # noqa: BLE001 - user supplied callback boundary
            # A reporting callback must not leave the audio device running.
            _LOG.exception("Microphone status callback failed")

    def _diagnostic(self, message: str) -> None:
        try:
            self._diagnostics.put_nowait(message)
        except queue.Full:
            pass

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        if self._stop.is_set():
            return
        if status:
            self._generation += 1
            self.dropped_chunks += 1
            # Older queued chunks precede the reported hardware gap.
            self._clear_audio()
            self._diagnostic(f"Audio gap: {status}. Detection state reset.")
        chunk = np.array(indata, dtype=np.float32, copy=True)
        try:
            self._audio.put_nowait((chunk, self._generation))
        except queue.Full:
            self.dropped_chunks += 1
            self._generation += 1
            self._clear_audio()
            self._diagnostic("Audio queue full: dropped audio. Detection state reset.")
            try:
                self._audio.put_nowait((chunk, self._generation))
            except queue.Full:
                pass

    def _clear_audio(self) -> None:
        while True:
            try:
                self._audio.get_nowait()
            except queue.Empty:
                return

    def _finished(self) -> None:
        if not self._stop.is_set():
            self.last_error = (
                "Microphone stream ended or disconnected. Select the microphone "
                "again and press Start."
            )
            self._diagnostic(self.last_error)
            self._stop.set()

    def start(self) -> None:
        """Open the chosen device at its native rate; do not silently switch."""
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self.running:
            return
        if self._worker and self._worker.is_alive():
            raise MicrophoneError("The previous microphone session is still stopping.")
        sd = _sounddevice()
        self._clear_audio()
        while not self._diagnostics.empty():
            try:
                self._diagnostics.get_nowait()
            except queue.Empty:
                break
        self._stop.clear()
        self._generation = 0
        self.dropped_chunks = 0
        self.last_error = None
        self.detector.reset()
        try:
            info = sd.query_devices(self.device, "input")
            channels = min(int(info["max_input_channels"]), 2)
            if channels < 1:
                raise ValueError("The selected device has no input channels")
            self.sample_rate = round(float(info["default_samplerate"]))
            if self.sample_rate < 1:
                raise ValueError("The device reported an invalid sample rate")
            sd.check_input_settings(
                device=self.device, channels=channels, dtype="float32", samplerate=self.sample_rate
            )
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=channels,
                dtype="float32",
                blocksize=0,
                callback=self._callback,
                finished_callback=self._finished,
            )
            self._running = True
            self._worker = threading.Thread(
                target=self._infer, name="HearSafe microphone inference", daemon=True
            )
            self._worker.start()
            self._stream.start()
        except Exception as exc:
            self._stop.set()
            self._running = False
            self._close_stream()
            if self._worker:
                self._worker.join(timeout=2)
            self.last_error = (
                f"Cannot open the selected microphone: {exc}. Check microphone "
                "permission, close other apps using it, or choose another device."
            )
            raise MicrophoneError(self.last_error) from exc
        self._status(f"Listening at {self.sample_rate} Hz. Raw audio is not saved.")

    def _infer(self) -> None:
        previous_generation = 0
        try:
            while not self._stop.is_set():
                while True:
                    try:
                        self.detector.set_alerts(self._controls.get_nowait())
                    except queue.Empty:
                        break
                while True:
                    try:
                        self._status(self._diagnostics.get_nowait())
                    except queue.Empty:
                        break
                try:
                    chunk, generation = self._audio.get(timeout=0.1)
                except queue.Empty:
                    continue
                if generation != previous_generation:
                    self.detector.reset()
                    previous_generation = generation
                self.on_level(float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64))))
                for frame in self.detector.process(chunk, self.sample_rate):
                    if self._stop.is_set():
                        break
                    self.on_frame(frame)
        except Exception as exc:  # noqa: BLE001 - detector/runtime worker boundary
            self.last_error = (
                f"Microphone processing stopped: {exc}. Select the microphone and restart."
            )
            self._status(self.last_error)
        finally:
            self._stop.set()
            self._running = False
            while True:
                try:
                    self._status(self._diagnostics.get_nowait())
                except queue.Empty:
                    break
            self._close_stream()
            try:
                self.detector.reset()
            except Exception:  # noqa: BLE001 - cleanup must release the device
                _LOG.debug("Detector reset during cleanup failed", exc_info=True)

    def _close_stream(self) -> None:
        with self._stream_lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop()
                except Exception:  # noqa: BLE001 - disconnected device cleanup
                    _LOG.debug("Microphone stop during cleanup failed", exc_info=True)
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - disconnected device cleanup
                    _LOG.debug("Microphone close during cleanup failed", exc_info=True)

    def stop(self) -> None:
        """Stop capture promptly and discard unfinished microphone windows."""
        with self._lifecycle_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self._stop.set()
        self._running = False
        self._close_stream()
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=3)
        self._clear_audio()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
