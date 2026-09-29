# Portability and integration plan

HearSafe should work as a reusable sound detector with optional interfaces around it. A desktop application is one client of the detector. A hardware project can use the same detector with its own microphone and display.

This document describes the intended design. These components have not been implemented yet.

## Components

```text
Microphone, WAV file, or caller-provided audio
                     |
                     v
             Audio input adapter
                     |
                     v
      Shared preprocessing and buffering
                     |
                     v
            Local model inference
                     |
                     v
       Event filter and alert policy
                     |
          +----------+----------+
          |          |          |
     Python API   JSON CLI   Desktop UI
```

The detector must accept audio supplied by another program. It must not require the desktop window or direct access to a particular microphone. Microphone selection and capture belong in replaceable input adapters.

## Proposed input contract

- Mono PCM audio, represented as `float32` samples in the range `[-1, 1]`.
- The caller provides the sample rate and capture timestamp with each chunk.
- The input adapter converts device audio to the model's documented sample rate. The initial target is 16 kHz.
- The detector accepts consecutive chunks and handles buffering and overlapping analysis windows internally.
- It can operate offline on a WAV file and online on live microphone chunks using the same preprocessing path.

## Proposed event contract

Return a structured event rather than a UI notification. An example:

```json
{
  "schema_version": 1,
  "label": "door_knock",
  "score": 0.91,
  "time_ms": 1730000000000,
  "state": "started"
}
```

`score` is a model score, not a guarantee that the event occurred. The event filter should implement per-class thresholds, repeated-evidence checks, and cooldowns. The consuming program chooses whether to show a visual alert, vibrate, log an event, or take no action.

## Distribution targets

1. **Python package:** expose a small API for feeding audio chunks and receiving events. Keep training dependencies separate from runtime dependencies.
2. **Command-line tool:** accept WAV files or a selected microphone and optionally emit newline-delimited JSON events. This lets other programs integrate without importing Python code.
3. **Desktop application:** use the same detector and bundle a CPU runtime, model, and UI as a downloadable folder or executable. Build and test a package separately for each operating system.
4. **Model package:** export a small model to ONNX if accuracy and preprocessing parity hold. Include labels, configuration, model version, license information, and test audio with expected outputs. ONNX Runtime offers Python and C APIs and CPU execution on several desktop and edge platforms. Hardware-specific support must be verified on the actual target device.

The first release target should be a normal Windows or Linux computer with a USB or built-in microphone. Small Linux boards can follow once CPU use, memory, latency, and microphone capture are measured. Microcontrollers require a separate feasibility assessment because their memory and runtime support differ substantially.

## Acceptance checks

- The same recorded audio produces matching labels and scores through the Python API, JSON CLI, and desktop app.
- A caller can supply audio without installing or opening the desktop UI.
- The runtime operates locally, with no network connection required for detection.
- The model's sample rate, window length, labels, thresholds, and license are included in the model package.
- On target hardware, report detection delay, CPU use, memory use, missed events, and false alerts.
- A model update cannot silently change the event schema or label meanings.

## Reference documentation

- [ONNX Runtime Python API](https://onnxruntime.ai/docs/get-started/with-python.html)
- [ONNX Runtime C API](https://onnxruntime.ai/docs/get-started/with-c.html)
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)
- [python-sounddevice input streams](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)
- [PyInstaller bundle behavior](https://pyinstaller.org/en/stable/operating-mode.html)
