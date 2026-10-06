"""Stateful streaming detector, independent of microphones and desktop windows."""

from time import perf_counter

import numpy as np
import soxr

from .alerts import AlertFilter
from .audio import mono_audio
from .types import AlertRule, AnalysisFrame, Prediction


class Detector:
    def __init__(self, model, alert_config: dict[str, AlertRule] | None = None):
        self.model = model
        self._filter = AlertFilter()
        self.set_alerts(alert_config or {})
        self.reset()

    def set_alerts(self, rules: dict[str, AlertRule]):
        valid = {label.id for label in self.model.manifest.labels}
        if set(rules) - valid:
            raise ValueError("Watchlist contains labels not present in the selected model")
        self._filter = AlertFilter(rules)

    def reset(self):
        self._buffer = np.empty(0, dtype=np.float32)
        self._rate = None
        self._resampler = None
        self._start = 0
        self._received = 0
        self._last_covered = 0
        self._finished = False
        self._filter.reset()

    def process(self, audio, sample_rate: int) -> list[AnalysisFrame]:
        if self._finished:
            raise ValueError("Call reset() before processing a new audio session")
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer")
        data = mono_audio(audio)
        if not len(data):
            return []
        if self._rate is None:
            self._rate = int(sample_rate)
            if self._rate != self.model.manifest.sample_rate:
                self._resampler = soxr.ResampleStream(
                    self._rate, self.model.manifest.sample_rate, 1, dtype="float32", quality="HQ"
                )
        elif self._rate != sample_rate:
            raise ValueError("Sample rate changed within a session; call reset() first")
        if self._resampler:
            data = np.clip(self._resampler.resample_chunk(data), -1, 1)
        return self._append(data)

    def _append(self, data):
        self._received += len(data)
        self._buffer = np.concatenate((self._buffer, data))
        result = []
        window = self.model.manifest.window_samples
        hop = self.model.manifest.hop_samples
        while len(self._buffer) >= window:
            result.append(self._infer(self._buffer[:window], self._start + window))
            self._last_covered = self._start + window
            self._buffer = self._buffer[hop:]
            self._start += hop
        return result

    def _infer(self, window, actual_end):
        started = perf_counter()
        scores = np.asarray(self.model.predict(window), dtype=np.float32).reshape(-1)
        duration = (perf_counter() - started) * 1000
        if (
            len(scores) != len(self.model.manifest.labels)
            or not np.isfinite(scores).all()
            or np.any(scores < 0)
            or np.any(scores > 1)
        ):
            raise ValueError("Model returned invalid scores or mismatched labels")
        predictions = [
            Prediction(label.id, label.name, float(score))
            for label, score in zip(self.model.manifest.labels, scores, strict=True)
        ]
        rate = self.model.manifest.sample_rate
        start_ms, end_ms = self._start * 1000 / rate, actual_end * 1000 / rate
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
        events = self._filter.process(predictions, self.model.manifest.model_id, end_ms, rms)
        return AnalysisFrame(
            self.model.manifest.model_id, start_ms, end_ms, predictions, rms, duration, events
        )

    def flush(self) -> list[AnalysisFrame]:
        if self._finished:
            return []
        result = []
        if self._resampler is not None:
            data = np.clip(
                self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True), -1, 1
            )
            result.extend(self._append(data))
        # Predict a padded final window only if it covers previously unanalysed samples.
        if self._received > self._last_covered and len(self._buffer):
            window = np.pad(
                self._buffer, (0, self.model.manifest.window_samples - len(self._buffer))
            )
            result.append(self._infer(window, self._received))
        self._finished = True
        return result
