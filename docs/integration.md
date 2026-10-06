# Use HearSafe inside another project

HearSafe runs on the computer or board connected to your microphone. The model
uses CPU inference locally. The desktop app, file commands, microphone commands,
and Python API use the same detector, so a hardware project can choose its own
microphone adapter and use the resulting sound labels to drive a display, light,
or other action.

The first portable bundle is Windows x64. Linux and Raspberry Pi are source
integration targets that need measurements on the actual device; this Windows
ZIP does not run on them. A microphone alone cannot run the model. Small
microcontrollers need a different deployment and model feasibility assessment.
The portable bundle uses standard Windows input backends such as WASAPI;
optional ASIO devices require a separate source deployment.

## Python setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From a source checkout:

```powershell
uv sync --python 3.12
uv run --frozen hearsafe download-model
uv run --frozen hearsafe devices
```

The runtime installs CPU inference and audio dependencies. Training is optional:
`uv sync --extra train`. Desktop and embedded detection do not require PyTorch.
Model downloads happen during setup; detection then works offline.

## Supply audio from your own program

```python
import soundfile as sf
from hearsafe import Detector, load_model

detector = Detector(load_model("models/yamnet"))
with sf.SoundFile("test.wav") as source:
    for audio in source.blocks(blocksize=4096, dtype="float32", always_2d=True):
        for frame in detector.process(audio, source.samplerate):
            print(frame.to_dict())
for frame in detector.flush():
    print(frame.to_dict())
```

`process(audio, sample_rate)` accepts consecutive NumPy float audio samples in
the range `[-1, 1]`, either mono or samples-by-channels stereo. Supply the actual
sample rate. The detector mixes channels, resamples continuously, buffers
overlapping windows, and returns zero or more analysis frames. Float audio is
the integration contract: convert integer PCM to normalized floats first.

Call `flush()` once at the end of a file to analyse its remaining samples with
padding. Do not flush between live chunks. Call `reset()` after an input gap,
device change, or sample-rate change. Construct a new detector when changing
models; model state and audio must not carry across the switch. Keep one detector
on one inference worker, because its buffering state is mutable.

An analysis frame exposes `.to_dict()` for JSON serialization. Its versioned
output includes model identity, audio-relative times, predictions with stable
label identifiers and model scores, and selected alert events. Scores are not
accuracy percentages. YAMNet categories may overlap; the research model uses
the 50 ESC-50 categories and has a different score interpretation. Treat label
IDs together with the model identifier as the integration key.

YAMNet needs its first 15,600 samples at 16 kHz (0.975 seconds), then analyses
every 0.48 seconds. The research model needs five seconds of audio first and
also advances by 0.48 seconds. Device buffers and CPU work add to observed
delay. Live top candidates are model output; alerts additionally apply the
watchlist's thresholds and timing rules. Alerts begin disabled.

`examples/embed_wav.py` demonstrates file integration. `examples/embed_microphone.py`
shows a blocking input worker suitable for adapting to another application's
worker loop. GUI applications should keep capture, inference, and UI updates on
separate workers. Audio callbacks must not run inference or write to disk.

## Call the portable CLI from another language

PowerShell, after extracting the whole portable folder:

```powershell
.\hearsafe-cli.exe devices
.\hearsafe-cli.exe detect "C:\audio\sound.wav" --json
.\hearsafe-cli.exe listen --device 0 --json
.\hearsafe-cli.exe benchmark --iterations 50 --paced
```

Use the integer microphone ID printed by `devices`. Use `--model "C:\models\my-model"`
to load a compatible model package. During development, replace
`.\hearsafe-cli.exe` with `uv run --frozen hearsafe`.

`--json` emits one JSON object per analysis frame on standard output. Parse each
line independently, check `schema_version`, and keep standard error separate:
diagnostics are written there. Interrupt `listen` with Ctrl+C to stop.
`examples/json_subprocess.py` shows how to launch the command safely with an
argument list, consume complete lines, and clean up the process.

By default, models are loaded from `models/yamnet` beside the portable
executables. `--model` chooses a package explicitly; `HEARSAFE_MODEL_DIR` changes
the default location. Keep the entire ZIP's extracted folder together when
moving it. Neither the desktop app nor CLI needs the source checkout or Python
installed on the destination computer.

## Model packages and custom hardware

A model package contains its manifest, labels, model file, preprocessing
configuration, checksum, license, and provenance. The loader validates the
package before capture starts. The bundled model uses LiteRT CPU; locally
trained research models use ONNX Runtime CPU with shared preprocessing.
The portable app loads research weights without installing PyTorch.

Begin on the hardware with a known WAV file and benchmark. Then connect its
microphone, log sound events, and measure CPU use, peak memory, processing time,
misses, and false alerts using `docs/testing.md`. CPU inference should remain
below the 0.48-second hop interval so the queue stays bounded. CPU architecture
and runtime wheel availability must be checked before promising a Linux board.

The source code is MIT licensed. Model and audio licenses are separate. ESC-50
audio and models trained from it remain local in this prototype; review the
dataset's noncommercial terms before distributing those weights.
