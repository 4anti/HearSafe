# Prototype measurements

Measured on 2026-10-06 on Windows 11 x64, Intel Core i9-13980HX, with Python
3.12.12 and locked CPU dependencies. These are build-computer measurements;
they do not establish performance on a smaller computer, Linux, or Raspberry Pi.

## Sound explorer

The official YAMNet LiteRT package contains 521 categories and a 4,126,810-byte
model. Its model SHA256 is
`4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`.

The paced source benchmark ran 64 windows, one every 480 ms, over 30.76 seconds:

| Measurement | Result |
| --- | --- |
| Median model inference | 4.13 ms |
| 95th percentile inference | 5.07 ms |
| Maximum inference | 5.45 ms |
| Average CPU, 100% = one logical CPU | 1.57% |
| Observed resident memory | 102.23 MiB |
| Process peak working set on Windows | 106.81 MiB |
| Model loading | 36.4 ms |
| Meets the 480 ms hop at p95 | Yes |

See [the machine-readable benchmark](yamnet_benchmark.json). This uses synthetic
model-rate audio and includes model inference, not microphone capture,
resampling, or the desktop interface. Other build/training processes were
running on the machine. It is a 30-second benchmark, not the 30-minute room test.

## Six file examples and weak observations

The first fold-1 ESC-50 file for each of six selected categories was analysed,
without choosing examples based on the output. Peak matching scores were:

| File category | Matching YAMNet category | Peak score |
| --- | --- | --- |
| Dog | Dog | 0.9414 |
| Wood knock | Knock | 0.8008 |
| Alarm clock | Alarm clock | 0.6680 |
| Siren | Siren | 0.5859 |
| Baby crying | Baby cry, infant cry | 0.0820 |
| Glass breaking | Shatter | 0.1484 |

Baby crying and glass breaking were weak in these examples: even the broader
`Crying, sobbing` (0.1094) and `Breaking` (0.4141) categories stayed below 0.5.
The baby-cry example also produced laughter candidates. Related categories can
overlap: the alarm example also produced telephone candidates, and broad animal
categories outranked Dog in the dog example. A peak above threshold alone does
not prove that the sustained alert rules will fire.

See [the six-example report](yamnet_file_smoke.json) for filenames, audio/model
checksums, frame times, top five, and reproduction commands. This is a file
smoke check, not category accuracy or speaker-to-microphone validation. No
dataset audio is copied into the report.

## HearSafe research CNN

The full seeded CPU run completed 50 epochs. Epoch 45 was selected using
fold-4 validation loss (1.3051), then evaluated once on fold 5. Folds 1–3 contain
1,200 training clips; fold 4 contains 400; fold 5 contains 400, eight per class.
The training/export/evaluation run took about 18 minutes 40 seconds.

| Held-out-fold result | Measurement |
| --- | --- |
| Accuracy on 400 fold-5 clips | 62.0% |
| Macro F1 across all 50 categories | 0.5980 |
| ONNX model size | 122,256 bytes |
| PyTorch/ONNX maximum export logit difference | 0.00000477 |
| First analysis window / update step | 5 s / 0.48 s |

This is a held-out **clip-fold** experiment, not five-fold cross validation,
source-independent testing, or measured microphone accuracy. The supplied
metadata has four source identifiers shared between official folds:
`131943` and `134049` across folds 2/3; `209698` and `234879` across folds 4/5.
The latter two mean validation and test clips share original source identifiers.
The prescribed folds were preserved and this limitation is reported openly.

Known weak categories on fold 5 include helicopter and fireworks (0/8 recalled),
wood creaks, clapping, and crackling fire (1/8), and pig (2/8). Relevant trial
categories performed as follows: wood knock and clock alarm 7/8, baby crying
6/8, siren and glass breaking 5/8, and dog 3/8. Eight clips per category is a small
sample; these are classifier recalls, not alert-policy results.

Full [per-class results](esc50_evaluation.md), [confusion matrix and metrics](esc50_evaluation.json),
[training configuration/history](esc50_training.json), and
[model manifest snapshot](esc50_model_manifest.json) are published. Weights and
dataset audio stay local under the dataset terms. The selected model folder can
be loaded in the desktop application without PyTorch.

The fresh paced runtime benchmark ran 64 windows over 30.76 seconds:

| CPU runtime measurement | Result |
| --- | --- |
| Median frontend and inference | 12.41 ms |
| 95th percentile | 15.32 ms |
| Maximum | 17.25 ms |
| Average CPU, 100% = one logical CPU | 3.96% |
| Observed resident memory | 124.92 MiB |
| Process peak working set on Windows | 129.10 MiB |
| Model loading | 231.1 ms |
| Meets the 480 ms hop at p95 | Yes |

See [the selected-model benchmark](esc50_benchmark.json). It includes shared mel
preprocessing and inference, excluding capture, resampling, file I/O, and UI.

## CPU configuration

Paced testing caught high ONNX Runtime CPU use while waiting between windows.
The runtime now disables intra/inter-op idle spinning, following the official
[thread management guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
In a preliminary research-model check, average CPU fell from about 99% to 6.5%
of one logical CPU while p95 inference stayed below 17 ms. The final selected
weights are benchmarked separately below. This changes runtime scheduling,
not the model's training or score interpretation.

## Automated and portable checks

43 automated tests pass. Coverage includes irregular chunks at 16/44.1/48 kHz,
stereo mixing, short-file padding, silence, invalid audio, corrupt WAVs,
checksum/manifest errors, brief and sustained alerts, overlapping categories,
cooldown/rearming, capture gaps/overflow, denied/disconnected microphones, and
model switches. Real YAMNet results match between the Python API, JSON CLI,
and desktop file worker within an absolute score tolerance of 0.000001.
Export tests verify PyTorch/ONNX logits and embedded training normalization.

The Windows bundle is tested from a copied folder containing spaces, with
source/model location environment variables removed. Both executable startup
and WAV inference are checked. Research ONNX loading is checked in the bundled
runtime, which excludes PyTorch. The public ZIP includes only YAMNet weights,
runtime dependencies, and license notices; training source stays in the repo.

The final frozen executables were also benchmarked for 30 paced windows each:

| Bundled runtime | p95 inference | CPU, one-core basis | Peak working set |
| --- | --- | --- | --- |
| YAMNet | 5.79 ms | 1.30% | 114.53 MiB |
| Selected research ONNX | 9.03 ms | 3.14% | 137.37 MiB |

See [the portable smoke report](portable_smoke.json). These runs use two-second
silence for the loading check and synthetic audio for benchmarking. The shorter
benchmark and system load differ from the 64-window source runs above. The
research check loads local weights outside the public ZIP.

## Checks requiring hardware trials

The following results are **not measured yet**:

- Recognition accuracy or alert delay through the HyperX headset microphone.
- Ten trials per target sound at 0.5 m and 2 m, with processing settings recorded.
- Misses, wrong alerts, and duplicates during a 30-minute background session.
- A separate Windows computer without Python, internet, or administrator rights.
- Linux and Raspberry Pi inference/capture performance.

Use [the headset test procedure](../docs/testing.md) and export the trial notes
alongside application results. The application does not save microphone audio.
Scores, dataset accuracy, and file smoke examples cannot replace those trials.
