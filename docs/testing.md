# Test HearSafe with your headset

You can test sounds you make or play through a phone or speaker. Your microphone
must actually receive the sound: playback only inside your headphones may be
too quiet outside the ear cups. Recognition is limited to the selected model's
categories, and an imitation can get a different label from the original sound.

## First microphone test

1. Extract the entire portable ZIP, open `HearSafe.exe`, and refresh microphones.
2. Select your headset's input. Use the Windows microphone privacy settings to
   allow desktop apps if Windows blocks it. Check another recording app if the
   selected device is busy or does not provide audio.
3. Choose the bundled Sound explorer model and click Start. Speak at a normal
   volume and check that the input meter moves. YAMNet collects 0.975 seconds
   before its first prediction. The locally trained model needs five seconds.
4. Clap, whistle, speak, and knock on a table while watching the top candidates.
   Compare repeated attempts. Scores show model output, not the probability of
   a correct real-world alert.
5. Select a few matching categories in the watchlist, set their thresholds,
   then enable alerts. Begin with the default settings and log results before
   adjusting them.

Use a steady microphone position and avoid clipping. Keep the phone volume
comfortable; realistic tests do not require loud sounds. Headsets often suppress
background sounds to make speech clearer. Compare your headset's noise
suppression and Windows audio-enhancement settings when they are available.
Record the settings for each run so results can be reproduced. If useful sounds
vanish, repeat with a basic USB or built-in microphone.

## Controlled sound trials

Use recorded alarms, sirens, dogs, baby crying, and glass breaking, plus actual
claps and knocks. Test direct WAV files first to establish the model's response.
Then play those same files from a phone or external speaker so the test includes
your room, microphone, playback volume, and headset processing.

For each sound, run ten trials at approximately 0.5 metres and ten at 2 metres.
Leave enough silence between trials for the event to rearm. Record:

| Field | What to write |
| --- | --- |
| Model and version | The model shown in the application |
| Device and settings | Microphone, processing settings, playback source, volume |
| Actual sound | The expected sound and file or action used |
| Distance | 0.5 m or 2 m |
| Result | Correct candidate, wrong candidate, or missed sound |
| Alert | Correct alert, wrong alert, no alert, or duplicate alert |
| Delay | Seconds between starting the sound and the first useful result |
| Notes | Background noise, clipping, or interrupted capture |

Export event history to keep the application's timestamps and scores. Exported
events cannot identify missed sounds by themselves; keep the trial log as well.
This prototype does not record microphone audio. Use a separate recording
program only when you explicitly choose to record test trials, and include
the processing settings in its notes.

## Thirty-minute background check

Enable the same watchlist and thresholds, then run for 30 minutes with ordinary
speech, typing, music, and room noise. Count unexpected alerts and repeated
alerts for a single sound. Combine this with the controlled trial log to count
missed target sounds. Do not tune thresholds on the same trials you later claim
as an independent evaluation. Make changes, then collect a new test run.

Neither a good file score nor one successful headset trial establishes reliable
performance in another room. This prototype is an assistive awareness tool and
does not replace an emergency alarm.

## Developer checks

From a source checkout with Python 3.12:

```powershell
uv sync --extra dev --extra build
uv run --frozen --extra dev pytest
uv run --frozen hearsafe download-model
uv run --frozen hearsafe detect "test.wav" --json
uv run --frozen hearsafe benchmark --iterations 50 --paced
uv run --frozen --extra build python scripts/build_portable.py
```

The build creates `dist/HearSafe-Windows-x64.zip` and a SHA256 sidecar. It smoke
tests desktop startup, CLI commands, and inference from a relocated folder containing spaces,
with source-path and model-location environment variables removed. Its JSON
results are included in the bundle. Fixed ZIP ordering and timestamps stabilize
archive metadata; this is not a claim that executable builds are reproducible
across computers or dependency versions.

Automated coverage checks irregular chunks and rates, stereo mixing, final
padding, silence, invalid input, event timing/rearming, and model integrity.
Training checks verify the fixed held-out-fold split, training-only
normalization, and PyTorch-to-ONNX parity. Runtime packages must load exported
models without PyTorch.

Run the ZIP on a separate Windows x64 computer without Python, internet access,
or administrator rights. Select its microphone, confirm the meter and sound
predictions, then move the extracted folder and repeat a known WAV test. This
manual check remains necessary even when the build-computer smoke test passes.
Windows builds must be produced on Windows; other operating systems need their
own build and validation.

Record the CPU, Windows version, app version, model version, average inference
time, slowest measured inference time, CPU use, and peak memory for both models.
The processing time should stay below the 0.48-second analysis interval with no
growing backlog. Tests must also exercise a disconnected or busy microphone,
input queue overflow, model switching, and a missing or incompatible model.
The interface should stop capture with a useful message after a device failure.

## Research-model results

Train using local ESC-50: folds 1–3 are training, fold 4 selects the checkpoint,
and fold 5 is final evaluation. Report accuracy, macro F1, per-class recall,
confusion matrix, model size, and CPU inference time. Call this a held-out-fold
experiment; it is not five-fold cross validation or measured headset accuracy.
Keep dataset audio and trained weights out of the public portable ZIP.

```powershell
uv sync --extra train --extra dev
uv run --frozen --extra train hearsafe train --dataset data/ESC-50-master --output data/models/esc50
uv run --frozen --extra train hearsafe evaluate --dataset data/ESC-50-master --model data/models/esc50 --fold 5
```

The training command writes its manifest, ONNX model, best checkpoint, history,
and reports into the selected local output folder. Load that model folder in
the desktop application to compare it with Sound explorer.
To verify that the portable app can load those local weights without PyTorch,
rebuild with `uv run --frozen --extra build python scripts/build_portable.py --research-model data/models/esc50`.
This adds a smoke check and benchmark; the research weights stay outside the ZIP.

Published prototype results should identify tested hardware, the exact model,
weak categories, and any checks still awaiting a second computer. Avoid making
an accuracy claim without its dataset or real microphone testing conditions.
