"""One audio conversion and feature path shared by training, files and live input."""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy.signal.windows import hann


def mono_audio(audio) -> np.ndarray:
    data = np.asarray(audio)
    if data.dtype.kind not in "f":
        raise ValueError("Audio must contain floating point PCM samples in [-1, 1]")
    if data.ndim == 2 and data.shape[1] >= 1:
        data = data.mean(axis=1)
    if data.ndim != 1:
        raise ValueError("Audio must have shape [samples] or [samples, channels]")
    if not np.isfinite(data).all():
        raise ValueError("Audio contains NaN or infinite samples")
    if data.size and np.max(np.abs(data)) > 1.0001:
        raise ValueError("Audio values must be in [-1, 1]")
    return np.asarray(data, dtype=np.float32)


def load_wav(path: Path | str) -> tuple[np.ndarray, int]:
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError(f"Cannot read audio file: {exc}") from exc
    return mono_audio(data), int(rate)


def iter_wav_chunks(path: Path | str, chunk_size: int = 4096):
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    try:
        with sf.SoundFile(str(path)) as file:
            rate = int(file.samplerate)
            while True:
                data = file.read(chunk_size, dtype="float32", always_2d=True)
                if not len(data):
                    break
                yield mono_audio(data), rate
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError(f"Cannot read audio file: {exc}") from exc


def resample_audio(audio, from_rate: int, to_rate: int = 16000) -> np.ndarray:
    data = mono_audio(audio)
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if not len(data) or from_rate == to_rate:
        return data.copy()
    converted = soxr.resample(data, from_rate, to_rate, quality="HQ")
    # Bandlimited resampling can overshoot valid PCM slightly; keep runtime/training identical.
    return np.clip(converted, -1, 1).astype(np.float32)


def _mel_filter() -> np.ndarray:
    # HTK mel scale, matching the manifest's fixed frontend specification.
    low, high = 2595 * np.log10(1 + np.array([125.0, 7500.0]) / 700)
    points = 700 * (10 ** (np.linspace(low, high, 66) / 2595) - 1)
    frequencies = np.fft.rfftfreq(512, 1 / 16000)
    lower = (frequencies[:, None] - points[:-2]) / (points[1:-1] - points[:-2])
    upper = (points[2:] - frequencies[:, None]) / (points[2:] - points[1:-1])
    return np.maximum(0, np.minimum(lower, upper)).astype(np.float32)


_MEL = _mel_filter()
_HANN = hann(400, sym=False).astype(np.float32)


def log_mel(audio) -> np.ndarray:
    data = mono_audio(audio)
    if len(data) != 80000:
        raise ValueError("ESC-50 frontend expects exactly 80,000 samples at 16 kHz")
    frames = np.lib.stride_tricks.sliding_window_view(data, 400)[::160]
    spectrum = np.abs(np.fft.rfft(frames * _HANN, n=512, axis=1)).astype(np.float32)
    return np.log(spectrum @ _MEL + 0.001).T.astype(np.float32)
