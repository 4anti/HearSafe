# HearSafe

An open-source, offline sound tester for microphones and WAV files. HearSafe runs
on your CPU: no account, paid API, GPU, or subscription is required. Other
projects can feed their own audio to the same detector without opening the app.

**Status: Windows prototype.** Sound recognition depends on the model, room,
microphone, and audio processing. This is an experimental awareness tool, not an
emergency alarm. Model scores are not accuracy percentages.

## Start with the portable app

1. Download `HearSafe-Windows-x64.zip` from [Releases](https://github.com/4anti/HearSafe/releases).
2. Extract the **whole folder**, then open `HearSafe.exe`. Keep the executables,
   `_internal` folder, and `models` folder together when moving it.
3. Select your headset microphone and the bundled **Sound explorer** model.
   Click **Start** and speak: the input meter should move.
4. Try clapping, whistling, speaking, and knocking. Watch the top five candidates.
   Play other sounds from a phone or room speaker so the microphone receives them.
5. Optionally add categories to the watchlist, set their thresholds, and enable
   visual alerts. Alerts start disabled. Export results and log your trials.

Detection works offline after extraction, without installing Python or using
administrator rights. Windows x64 is the first build target. See the
[headset testing guide](docs/testing.md) for distance trials, processing settings,
and the remaining second-computer acceptance check. No microphone audio is saved
by the app; it exports scores, events, and test notes.

## Two models

| Model | Categories | First analysis | Step | Score meaning |
| --- | --- | --- | --- | --- |
| Sound explorer: YAMNet | 521 AudioSet categories | 0.975 s | 0.48 s | Independent model scores; related categories can overlap |
| HearSafe research CNN | 50 ESC-50 categories | 5 s | 0.48 s | Softmax scores across these 50 classes |

The portable ZIP includes the official, checksum-verified YAMNet model and its
Apache 2.0 notices. It also supports loading the research model's local ONNX
folder without installing PyTorch. Train that model from your local ESC-50
dataset using the source workflow below; research weights and dataset audio are
excluded from public downloads.

Check [measured prototype results](reports/prototype_results.md) before making
claims about recognition quality or hardware requirements. Headset noise
suppression may hide sounds other than speech. An imitation may get a different
label, and sounds outside a model's categories cannot acquire a new label.

## Run from source

Use Python **3.12** and [uv](https://docs.astral.sh/uv/). The committed `uv.lock`
pins the tested dependency versions. From this folder:

```powershell
uv sync --frozen --python 3.12
uv run --frozen hearsafe download-model
uv run --frozen hearsafe gui
```

Only `download-model` downloads the model; detection never downloads it silently.
Runtime dependencies are separate from optional `train`, `dev`, and `build` extras.
Training uses the CPU PyTorch distribution.

## Train and evaluate the local research model

Extract [ESC-50](https://github.com/karolpiczak/ESC-50) into `data/ESC-50-master/`,
then run:

```powershell
uv sync --frozen --extra train --extra dev
uv run --frozen --extra train hearsafe train --dataset data/ESC-50-master --output data/models/esc50
uv run --frozen --extra train hearsafe evaluate --dataset data/ESC-50-master --model data/models/esc50 --fold 5
```

Use **Choose research folder…** in the desktop app and select `data/models/esc50`.
The training command writes ONNX weights, manifest, labels, training history,
accuracy, macro F1, per-class recall, confusion matrix, and performance reports.

This is a held-out-fold experiment: folds 1–3 train, fold 4 selects the checkpoint,
and fold 5 evaluates it. It is not five-fold cross validation or headset accuracy.
The fixed frontend uses 16 kHz mono, five-second windows, 64 mel bands, a
400-sample Hann window, 160-sample hop, and 512-point FFT. Normalization uses only
the training folds. The CNN has 16/32/64-channel blocks and a 50-class output;
defaults are seed 42, batch 32, AdamW 0.001, at most 50 epochs, and early stopping.

**Data terms:** ESC-50 is CC BY-NC; its ESC-10 subset is CC BY. HearSafe's MIT
license covers its own code and documentation. Review the dataset's
[license and attributions](https://github.com/karolpiczak/ESC-50#license) before
using or distributing audio or derived weights. Local data and research weights
are ignored by Git.

## Embed in another program

```python
from hearsafe import Detector, load_model

detector = Detector(load_model("models/yamnet"))
for audio in your_audio_chunks:  # normalized float mono or samples-by-channels
    for frame in detector.process(audio, sample_rate=48000):
        print(frame.to_dict())
for frame in detector.flush():  # end of a file; never between live chunks
    print(frame.to_dict())
```

`reset()` clears audio and alert state after a gap or device change. See the
[integration guide](docs/integration.md) and [examples](examples) for complete
microphone, file, and subprocess integrations. The portable CLI works from other
languages:

```powershell
.\hearsafe-cli.exe devices
.\hearsafe-cli.exe detect "sound.wav" --json
.\hearsafe-cli.exe listen --device 1 --json
.\hearsafe-cli.exe benchmark --iterations 50
```

Choose the device ID printed on **your** computer. JSON goes to standard output;
diagnostics go to standard error. Frames contain a schema version, model ID,
label IDs, model scores, audio-relative times, and optional selected alert events.

## Test and build

```powershell
uv sync --frozen --extra train --extra dev --extra build
uv run --frozen --extra train --extra dev pytest
uv run --frozen --extra build python scripts/build_portable.py
```

The native Windows build creates the ZIP and SHA256 sidecar in `dist/`. Build
other operating systems separately. Automated checks cover streaming rates and
chunk boundaries, file errors, alert rules, device failures, model validation,
and PyTorch/ONNX agreement. The build checks both executables in a relocated
folder with spaces. Real microphone trials and a separate Windows computer
without Python remain required release acceptance checks.

## Project documents and resources

- [Implementation roadmap](docs/roadmap.md)
- [Testing guide](docs/testing.md) and [integration guide](docs/integration.md)
- [Architecture and portability](docs/portability.md)
- [Prototype measurements](reports/prototype_results.md)
- [Original project notes](HearSafe_PROJECT_PLAN.md) and [class counts](reports/esc50_class_counts.md)
- [Google audio-classifier guide](https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier)
- [ESC-50 documentation](https://github.com/karolpiczak/ESC-50)
- [Windows audio processing](https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-signal-processing-modes)
- [PyInstaller packaging](https://pyinstaller.org/en/stable/operating-mode.html)

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute. HearSafe's original code and
documentation use the [MIT License](LICENSE); bundled third-party components
retain their own licenses and notices.
