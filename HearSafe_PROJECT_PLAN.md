# HearSafe — AI Environmental Sound Awareness System

> **Historical design notes:** The implemented Windows prototype and current scope are documented in [README.md](README.md) and [the roadmap](docs/roadmap.md). Earlier options and examples below are not current acceptance claims.
> **Primary goal:** Build an open-source, local-first AI system that recognizes important environmental sounds in real time and converts them into clear visual alerts.  
> **Target users:** Primarily deaf or hard-of-hearing users, but also anyone who benefits from visual awareness of surrounding sounds.  
> **Project type:** Audio Machine Learning + Signal Processing + Real-Time Inference + Desktop/Web Application  
> **Suggested repository name:** `HearSafe`

---

## 1. Project Summary

**HearSafe** is an AI-powered environmental sound recognition system.

The system continuously listens to microphone audio, analyzes short audio windows, recognizes relevant environmental events, and presents the detected event as a visual notification.

Example:

```text
Microphone
    │
    ▼
Audio Stream
    │
    ▼
Preprocessing
    │
    ├── resampling
    ├── mono conversion
    ├── normalization
    └── log-mel spectrogram
    │
    ▼
Audio Classification Model
    │
    ▼
Temporal Filtering + Confidence Threshold
    │
    ▼
┌──────────────────────────────────┐
│ HEARSAFE                         │
│                                  │
│ GLASS BREAKING DETECTED          │
│ Model score: 0.962               │
│                                  │
│ Detected 0.4 seconds ago         │
└──────────────────────────────────┘
```

The main idea is **not** to create another generic "classify this WAV file" notebook.

The goal is to create a complete, usable system:

```text
Dataset
   ↓
Training Pipeline
   ↓
Evaluation
   ↓
Exported Model
   ↓
Real-Time Audio Engine
   ↓
Application/API
   ↓
Visual Notifications
```

This makes the repository demonstrate several engineering skills at once:

- machine learning
- audio signal processing
- data engineering
- model evaluation
- real-time programming
- backend/API engineering
- application development
- testing
- documentation
- deployment

---

# 2. Problem We Are Trying to Solve

Many important events around a person are communicated primarily through sound.

Examples:

- door knocking
- alarms
- sirens
- crying baby
- glass breaking
- car horns
- dogs barking
- footsteps
- fire/crackling sounds
- thunder
- someone coughing
- a phone ringing

A person who cannot reliably hear those events may miss useful environmental information.

HearSafe attempts to transform:

```text
AUDIBLE INFORMATION
```

into:

```text
VISUAL INFORMATION
```

For example:

```text
Siren
  ↓
AI detection
  ↓
Red visual notification

Door knock
  ↓
AI detection
  ↓
Desktop notification

Baby crying
  ↓
AI detection
  ↓
Persistent alert

Glass breaking
  ↓
AI detection
  ↓
High-priority warning
```

---

# 3. What HearSafe Is NOT

The project should avoid making claims that are impossible to guarantee.

HearSafe should **not** initially be presented as:

- a certified emergency system
- a replacement for smoke detectors
- a medical device
- a police/security monitoring system
- a guaranteed life-safety system

Instead:

> HearSafe is an assistive environmental sound-awareness tool.

This distinction is important because machine-learning classifiers can produce:

- false positives
- false negatives
- low-confidence predictions
- failures in noisy environments
- failures with microphones different from the training data

The README should clearly state this.

---

# 4. Core MVP

The first version should be deliberately small.

Do **not** attempt to recognize hundreds of sounds immediately.

The initial MVP should recognize approximately **8–12 important classes**.

Recommended first classes:

| Priority | Sound |
|---|---|
| High | Glass breaking |
| High | Siren |
| High | Alarm |
| High | Baby crying |
| Medium | Door knock |
| Medium | Car horn |
| Medium | Dog barking |
| Medium | Footsteps |
| Medium | Fire / crackling fire |
| Medium | Thunderstorm |
| Optional | Coughing |
| Optional | Vacuum / loud appliance |

A special additional class should be created:

```text
OTHER / UNKNOWN
```

This is important.

Without an unknown/background class, a classifier may be forced to call every random noise:

```text
glass breaking
siren
baby crying
...
```

even when none of those sounds are present.

---

# 5. MVP Success Criteria

Version `0.1.0` should be considered successful when the project can:

- read `.wav` files
- convert them into a standardized representation
- generate log-mel spectrograms
- train a neural network
- evaluate that model on unseen data
- save model weights
- load the trained model for inference
- analyze microphone input
- produce predictions continuously
- filter unstable predictions
- display desktop or web notifications
- provide confidence scores
- run locally without sending microphone audio to a cloud service

Example CLI:

```bash
hearsafe detect example.wav
```

Output:

```text
[HearSafe]

Top prediction:
  Glass breaking       0.9421

Other candidates:
  Door knock           0.0284
  Alarm                0.0117
```

Real-time mode:

```bash
hearsafe listen
```

Output:

```text
[21:13:40] Background           0.91
[21:13:41] Background           0.88
[21:13:42] Door knock           0.94   ALERT
[21:13:43] Door knock           0.91
[21:13:44] Background           0.86
```

---

# 6. Dataset Strategy

The project should use datasets in stages.

Recommended strategy:

```text
PHASE 1
ESC-50
    ↓
small / easy / reproducible baseline

PHASE 2
UrbanSound8K
    ↓
additional urban/environmental variation

PHASE 3
FSD50K
    ↓
larger real-world dataset

PHASE 4
Custom recorded data
    ↓
microphone and room generalization

PHASE 5
Optional pretrained models / AudioSet
```

This is better than downloading a massive dataset immediately.

---

# 7. Dataset #1 — ESC-50

## Why We Should Start Here

ESC-50 is an excellent first dataset because it is:

- small
- clean
- labeled
- standardized
- commonly used in audio-classification research
- already divided into cross-validation folds
- easy to download
- easy to debug

Official repository:

https://github.com/karolpiczak/ESC-50

The dataset contains:

```text
2,000 audio recordings
50 sound classes
40 recordings per class
5 seconds per recording
44.1 kHz
mono
5 predefined folds
```

Examples of useful HearSafe classes already included in ESC-50:

- crying baby
- footsteps
- coughing
- door knock
- clock alarm
- glass breaking
- siren
- car horn
- dog
- crackling fire
- thunderstorm

Example directory:

```text
ESC-50/
├── audio/
│   ├── 1-100032-A-0.wav
│   ├── 1-100038-A-14.wav
│   └── ...
│
└── meta/
    └── esc50.csv
```

The metadata contains fields such as:

```text
filename
fold
target
category
esc10
src_file
take
```

## Important License Note

ESC-50 is distributed under:

```text
Creative Commons Attribution-NonCommercial
CC BY-NC
```

The smaller ESC-10 subset uses CC BY.

For an open-source research/portfolio project this is generally useful, but the non-commercial restriction must be respected.

Do not silently redistribute the entire dataset inside your Git repository.

Instead:

```text
scripts/download_esc50.py
```

or document how users can download it.

---

# 8. Dataset #2 — UrbanSound8K

Official information:

https://urbansounddataset.weebly.com/urbansound8k.html

Freesound dataset listing:

https://labs.freesound.org/datasets/

UrbanSound8K contains approximately:

```text
8,732 labeled clips
10 classes
clip duration <= 4 seconds
10 predefined folds
```

Classes include:

```text
air_conditioner
car_horn
children_playing
dog_bark
drilling
engine_idling
gun_shot
jackhammer
siren
street_music
```

For HearSafe, particularly useful classes are:

```text
car_horn
dog_bark
siren
```

UrbanSound8K is useful because its recordings come from actual urban recordings rather than controlled laboratory recordings.

That gives us additional environmental variation.

---

# 9. Dataset #3 — FSD50K

Companion website:

https://fsannotator.upf.edu/fsd/release/FSD50K/

Zenodo:

https://zenodo.org/records/4060432

FSD50K is much larger.

It contains approximately:

```text
51,197 audio clips
200 sound classes
100+ hours of audio
152,000+ ground-truth annotations
```

The dataset is based on Freesound recordings.

Important categories include:

- alarm
- bark
- crash
- vehicle sounds
- human sounds
- domestic sounds
- environmental sounds

FSD50K is useful for the second major generation of HearSafe because it contains much more variation than ESC-50.

Possible progression:

```text
ESC-50
   ↓
prove architecture works

FSD50K subset
   ↓
increase samples/class

FSD50K multi-label training
   ↓
support overlapping sounds
```

## Multi-Label Advantage

Real environments can contain several sounds simultaneously.

Example:

```text
traffic + siren + speech
```

A simple single-label classifier assumes:

```text
ONLY ONE CLASS IS CORRECT
```

FSD50K can help us later move toward:

```text
MULTI-LABEL CLASSIFICATION
```

Example prediction:

```json
{
  "siren": 0.91,
  "traffic": 0.78,
  "speech": 0.55
}
```

rather than:

```json
{
  "class": "siren"
}
```

## FSD50K Licensing

FSD50K contains clips with different Creative Commons licenses.

Examples include:

```text
CC0
CC-BY
CC-BY-NC
CC Sampling+
```

The dataset itself also has a dataset-level CC-BY license.

Because individual clips may have different restrictions, HearSafe should preserve attribution/license metadata when using these clips.

---

# 10. Dataset #4 — AudioSet

Official site:

https://research.google.com/audioset/

AudioSet is Google's large-scale audio-event dataset.

It contains roughly:

```text
2,084,320 labeled 10-second video segments
527 dataset labels
~5,800 hours of audio
```

AudioSet covers a very broad ontology:

```text
human sounds
animals
vehicles
alarms
tools
music
environmental noise
domestic sounds
mechanical sounds
...
```

AudioSet is extremely useful for pretrained models.

However:

AudioSet is based on YouTube segments.

That means it is less convenient than ESC-50/FSD50K for a beginner training pipeline.

Recommendation:

```text
DO NOT start HearSafe by downloading AudioSet.

USE models pretrained on AudioSet first.
```

---

# 11. Pretrained Model Option — YAMNet

Official TensorFlow example:

https://www.tensorflow.org/hub/tutorials/yamnet

Official model source:

https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

YAMNet predicts **521 audio-event classes** and was trained using AudioSet.

Architecture:

```text
Audio
  ↓
16 kHz mono
  ↓
STFT
  ↓
64-bin log-mel spectrogram
  ↓
MobileNetV1
  ↓
1024-dimensional embedding
  ↓
521 class probabilities
```

YAMNet processes roughly `0.96 second` patches.

This is valuable because HearSafe could use YAMNet in two different ways.

### Option A — Direct predictions

```text
Microphone
    ↓
YAMNet
    ↓
521 AudioSet classes
```

### Option B — Transfer learning

Better for our project:

```text
Microphone
    ↓
YAMNet feature extractor
    ↓
1024-D embedding
    ↓
Our custom classifier
    ↓
HearSafe classes
```

This allows us to train our own classification head.

Example:

```text
1024 input features

        ↓

Linear(1024, 256)
ReLU
Dropout

        ↓

Linear(256, 10)

        ↓

HearSafe classes
```

---

# 12. What Model Should We Actually Build?

To make the repository educational and prove that we understand ML, I recommend implementing **multiple model generations**.

---

## Model A — Classical Baseline

Purpose:

Establish a simple benchmark.

Features:

```text
MFCC
spectral centroid
spectral bandwidth
zero crossing rate
chroma
RMS energy
```

Model:

```text
Random Forest
or
SVM
```

Pipeline:

```text
audio.wav
   ↓
librosa
   ↓
handcrafted features
   ↓
SVM
   ↓
class
```

This gives us something to compare neural networks against.

---

## Model B — CNN From Scratch

This should be the primary educational model.

Pipeline:

```text
Waveform
   ↓
Log-Mel Spectrogram
   ↓
2D CNN
   ↓
Global Average Pooling
   ↓
Dense Layer
   ↓
Softmax
```

Example network:

```text
Input: [1, 64, 501]

Conv2D 1→32
BatchNorm
ReLU
MaxPool

Conv2D 32→64
BatchNorm
ReLU
MaxPool

Conv2D 64→128
BatchNorm
ReLU
MaxPool

Conv2D 128→256
BatchNorm
ReLU

Adaptive Average Pool

Linear 256→N_CLASSES
```

This is where we genuinely "train our own AI."

---

# 13. Why Log-Mel Spectrograms?

A waveform looks like this:

```text
Amplitude
   │
   │  /\     /\
   │ /  \   /  \
───┼─────\_/────\──────── Time
```

A spectrogram represents:

```text
frequency × time × intensity
```

Conceptually:

```text
Frequency
  ▲
8k│      ░░
6k│    ▒▒▒▒
4k│  ███▒▒
2k│████████
0k└────────────────────► Time
```

Different sounds have characteristic spectral patterns.

For example:

```text
SIREN
frequency repeatedly moves up/down

GLASS BREAK
short broadband high-frequency transient

FOOTSTEPS
repetitive low-frequency transients

BABY CRY
harmonic vocal structure

ALARM
repeating narrow-band tones
```

This makes spectrograms a natural representation for CNNs.

---

# 14. Proposed Audio Preprocessing

Recommended standard format:

```text
sample rate: 16,000 Hz
channels: mono
dtype: float32
range: [-1, 1]
clip length: 4–5 seconds
```

Pipeline:

```python
waveform
    ↓
convert stereo → mono
    ↓
resample → 16 kHz
    ↓
normalize
    ↓
pad/crop
    ↓
mel spectrogram
    ↓
log transformation
    ↓
normalization
```

Suggested mel parameters:

```text
sample_rate = 16000
n_fft = 1024
hop_length = 320
n_mels = 64
f_min = 50
f_max = 8000
```

These values can later be benchmarked.

---

# 15. Data Augmentation

ESC-50 only contains 40 examples per class.

That is small.

Data augmentation is therefore important.

Useful augmentation techniques:

### Gain augmentation

```text
quiet audio
   ↓
random volume
```

### Background noise

```text
door_knock.wav
+
room_noise.wav
```

### Time shift

```text
original:

-----SOUND-------

shifted:

--------SOUND----
```

### Time stretching

```text
0.90x
1.00x
1.10x
```

### Pitch shifting

Useful carefully.

```text
-2 semitones
+2 semitones
```

### SpecAugment

Mask portions of the spectrogram:

```text
████████████████
████░░░█████████
████░░░█████████
████████████████
```

This forces the network to rely on broader acoustic patterns.

---

# 16. Dataset Leakage — Important

Do not randomly split individual files without considering their original source.

ESC-50 already provides folds.

Use them.

Example:

```text
Fold 1 → Test
Fold 2 → Validation
Folds 3,4,5 → Training
```

Then rotate folds for proper cross-validation.

Why?

Because segments from the same original recording could otherwise appear in both training and test sets.

That would inflate performance.

---

# 17. Evaluation Metrics

Do **not** report only accuracy.

We should report:

```text
Accuracy
Precision
Recall
F1 Score
Macro F1
Confusion Matrix
Per-class Recall
Per-class Precision
```

Why recall matters:

Imagine:

```text
100 glass-breaking events

model detects only 60
```

Even if overall accuracy is high, that class performs poorly.

For an assistive system, missed events matter.

---

# 18. Confusion Matrix

Example:

```text
                  Predicted
                 Alarm Siren Glass Door
Actual Alarm       92     4     1     3
Actual Siren        5    89     2     4
Actual Glass        1     1    94     4
Actual Door         2     1     7    90
```

The repository should automatically generate:

```text
artifacts/
└── confusion_matrix.png
```

during evaluation.

---

# 19. Real-Time Detection Architecture

A microphone does not naturally produce neat 5-second WAV files.

We need streaming inference.

Pipeline:

```text
Microphone

   ↓ continuous audio

Ring Buffer

   ↓

1-second / 2-second windows

   ↓

Feature Extraction

   ↓

Model

   ↓

Prediction history

   ↓

Temporal smoothing

   ↓

Alert
```

Example prediction stream:

```text
t=0.0  siren=0.32
t=0.5  siren=0.77
t=1.0  siren=0.91
t=1.5  siren=0.95
t=2.0  siren=0.93
```

We should not trigger at:

```text
0.77
```

immediately.

Instead use temporal filtering.

---

# 20. Temporal Smoothing

Possible strategy:

Keep last `N` predictions.

Example:

```python
history = [
    0.81,
    0.87,
    0.92,
    0.94
]
```

Then calculate:

```text
mean confidence = 0.885
```

Alert only when:

```text
mean_confidence > threshold
AND
detections >= minimum_frames
```

Example:

```text
threshold = 0.80
minimum_frames = 3
```

This prevents notifications caused by very short false positives.

---

# 21. Alert Priority System

Not every detected sound should behave the same way.

Example:

```yaml
glass_breaking:
  priority: high
  cooldown: 10

siren:
  priority: high
  cooldown: 5

baby_crying:
  priority: medium
  cooldown: 10

door_knock:
  priority: medium
  cooldown: 3

dog:
  priority: low
  cooldown: 15
```

Application UI can render:

```text
HIGH
MEDIUM
LOW
```

without implying that the AI has certified an actual emergency.

---

# 22. Cooldown Logic

Without cooldowns:

```text
SIREN!
SIREN!
SIREN!
SIREN!
SIREN!
```

A siren lasting 30 seconds could trigger dozens of notifications.

Instead:

```text
event detected
     ↓
notification
     ↓
cooldown timer
     ↓
ignore same event temporarily
```

---

# 23. Proposed Repository Structure

```text
HearSafe/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── train.yaml
│   ├── model.yaml
│   └── alerts.yaml
│
├── hearsafe/
│   ├── __init__.py
│   │
│   ├── audio/
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   ├── features.py
│   │   ├── augment.py
│   │   └── stream.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── esc50.py
│   │   ├── fsd50k.py
│   │   └── labels.py
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   ├── baseline.py
│   │   └── yamnet.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   ├── smoothing.py
│   │   └── detector.py
│   │
│   ├── alerts/
│   │   ├── manager.py
│   │   └── desktop.py
│   │
│   └── cli.py
│
├── scripts/
│   ├── download_esc50.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   └── web/
│
├── tests/
│   ├── test_audio.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_stream.py
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_model_analysis.ipynb
│
├── assets/
│   ├── architecture.svg
│   ├── demo.gif
│   └── logo.svg
│
├── artifacts/
│   ├── confusion_matrix.png
│   └── metrics.json
│
└── docs/
    ├── datasets.md
    ├── model.md
    ├── training.md
    ├── inference.md
    └── roadmap.md
```

---

# 24. Dependencies

Recommended main stack:

```text
Python
PyTorch
torchaudio
librosa
numpy
pandas
scikit-learn
sounddevice
FastAPI
Pydantic
matplotlib
tqdm
PyYAML
```

Potential later dependencies:

```text
onnxruntime
TensorFlow / TensorFlow Hub
PyTorch Lightning
MLflow
Weights & Biases
Gradio
Electron / Tauri frontend
```

Keep version 0.1 relatively simple.

---

# 25. Configuration Instead of Hardcoding

Bad:

```python
SAMPLE_RATE = 16000
N_MELS = 64
NUM_CLASSES = 10
```

scattered across 10 files.

Better:

```yaml
audio:
  sample_rate: 16000
  clip_seconds: 5
  n_mels: 64
  n_fft: 1024
  hop_length: 320

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001

model:
  name: cnn
  num_classes: 10
```

This makes experiments reproducible.

---

# 26. Training Pipeline

Desired command:

```bash
python scripts/train.py --config configs/train.yaml
```

Pipeline:

```text
load config
    ↓
set random seed
    ↓
load metadata
    ↓
create datasets
    ↓
create dataloaders
    ↓
initialize model
    ↓
train
    ↓
validate
    ↓
save best checkpoint
    ↓
write metrics
```

Checkpoint structure:

```text
checkpoints/
├── best.pt
├── last.pt
└── metadata.json
```

Metadata should include:

```json
{
  "model": "HearSafeCNN",
  "dataset": "ESC-50",
  "sample_rate": 16000,
  "classes": [
    "alarm",
    "baby_cry",
    "car_horn"
  ],
  "epoch": 31,
  "validation_macro_f1": 0.91
}
```

---

# 27. Reproducibility

Set seeds:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Document:

```text
Python version
PyTorch version
CUDA version
GPU
dataset version
commit hash
configuration
```

Then another developer should be able to reproduce the results.

---

# 28. Benchmark File

Create:

```text
benchmarks/results.md
```

Example:

| Model | Input | Parameters | Accuracy | Macro F1 | CPU latency |
|---|---|---:|---:|---:|---:|
| SVM | MFCC | — | 71% | 0.69 | 8 ms |
| HearSafeCNN | Log-Mel | 1.8M | 88% | 0.87 | 18 ms |
| YAMNet + Head | Embedding | 3.7M+ | 94% | 0.93 | 31 ms |

Numbers above are examples only.

Never publish invented metrics.

Only add results generated by the actual benchmark script.

---

# 29. Training Experiments

Useful experiment progression:

### Experiment 001

```text
Dataset: ESC-50 subset
Model: CNN
Augmentation: none
```

### Experiment 002

```text
Dataset: ESC-50 subset
Model: CNN
Augmentation:
- random gain
- time shift
```

### Experiment 003

```text
Dataset: ESC-50 subset
Model: CNN
Augmentation:
- noise
- SpecAugment
```

### Experiment 004

```text
Model: pretrained embeddings
Classifier: custom MLP
```

Track:

```text
accuracy
macro F1
loss
per-class recall
training time
inference latency
```

---

# 30. Phase 1 Development Plan

## Milestone 1 — Dataset Explorer

Deliverables:

```text
scripts/download_esc50.py
notebooks/01_dataset_exploration.ipynb
```

Tasks:

- download ESC-50
- parse metadata
- select HearSafe classes
- count examples per class
- plot waveforms
- plot spectrograms
- listen to representative samples
- verify folds

Goal:

Understand the data before training.

---

# 31. Phase 2 — Audio Pipeline

Implement:

```text
loader.py
preprocess.py
features.py
```

Functions:

```python
load_audio()
resample_audio()
to_mono()
pad_or_crop()
create_log_mel()
```

Tests should verify:

```text
correct shape
correct sample rate
no NaNs
consistent output length
```

---

# 32. Phase 3 — Baseline Classifier

Implement classical model:

```text
MFCC → SVM
```

Reason:

If the neural network cannot beat a basic SVM, something may be wrong with the pipeline.

Deliver:

```text
baseline_metrics.json
baseline_confusion_matrix.png
```

---

# 33. Phase 4 — CNN

Implement:

```text
hearsafe/models/cnn.py
```

Train:

```bash
python scripts/train.py \
    --config configs/cnn_esc50.yaml
```

Requirements:

- GPU support
- checkpoint saving
- early stopping
- validation
- metrics logging
- deterministic seed

---

# 34. Phase 5 — Evaluation

Create:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt
```

Output:

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Per-class results
Confusion matrix
```

Example:

```text
Class            Precision   Recall   F1
------------------------------------------------
Glass Breaking      0.94      0.91   0.92
Siren               0.89      0.95   0.92
Door Knock          0.87      0.84   0.85
Baby Cry            0.96      0.92   0.94
```

---

# 35. Phase 6 — File Inference

Implement:

```bash
hearsafe detect path/to/file.wav
```

Optional JSON mode:

```bash
hearsafe detect sample.wav --json
```

Output:

```json
{
  "prediction": "siren",
  "confidence": 0.942,
  "top_k": [
    ["siren", 0.942],
    ["car_horn", 0.031],
    ["alarm", 0.014]
  ]
}
```

This makes the project easy to integrate into other software.

---

# 36. Phase 7 — Real-Time Microphone

Use:

```text
sounddevice
```

Architecture:

```text
InputStream
    ↓
callback
    ↓
ring buffer
    ↓
worker thread
    ↓
model inference
    ↓
event manager
```

Do model inference outside the raw audio callback.

The audio callback should remain lightweight.

---

# 37. Phase 8 — User Interface

First UI does not need to be complex.

Possible browser dashboard:

```text
┌────────────────────────────────────────────┐
│ HearSafe                         Listening │
├────────────────────────────────────────────┤
│                                            │
│          DOOR KNOCK                        │
│                                            │
│          Confidence 94%                    │
│                                            │
├────────────────────────────────────────────┤
│ Recent Events                              │
│                                            │
│ 21:07:32  Door Knock            94%        │
│ 21:03:14  Car Horn              87%        │
│ 20:58:02  Dog Bark              91%        │
└────────────────────────────────────────────┘
```

Useful settings:

```text
microphone selection
confidence threshold
enabled sounds
alert cooldown
notification volume
visual theme
history
```

---

# 38. Privacy

One major selling point should be:

> **Local-first inference.**

Microphone audio should not need to leave the device.

Ideal architecture:

```text
Microphone
   ↓
Local Model
   ↓
Local Alert
```

Not:

```text
Microphone
   ↓
Upload everything to server
   ↓
Cloud model
```

README feature:

```text
✓ Local inference
✓ No account required
✓ No continuous cloud recording
```

If logging clips for debugging becomes an optional feature, it should be disabled by default and clearly explained.

---

# 39. API

Later create:

```text
POST /predict
```

Example:

```bash
curl -X POST \
  -F "file=@door.wav" \
  http://localhost:8000/predict
```

Response:

```json
{
  "events": [
    {
      "label": "door_knock",
      "confidence": 0.94
    }
  ]
}
```

API stack:

```text
FastAPI
Pydantic
Uvicorn
```

---

# 40. Model Export

Long-term:

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime
```

Why?

ONNX allows easier deployment to:

- desktop applications
- Windows
- Linux
- edge devices
- different inference runtimes

Possible command:

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/best.pt
```

---

# 41. Performance Targets

Initial targets should be realistic.

Example project targets:

```text
Model size:
< 50 MB

CPU inference:
< 100 ms/window

GPU inference:
< 30 ms/window

Memory:
< 500 MB

Startup:
< 5 seconds
```

These are development goals, not guaranteed current results.

---

# 42. Hard Problem: Background Noise

Training audio often sounds cleaner than real microphone audio.

Real-world example:

```text
door knock
+
keyboard
+
fan
+
Discord
+
music
+
street traffic
```

The model needs exposure to these conditions.

Therefore later create custom augmentation:

```python
mixed_audio = target_sound + alpha * background_noise
```

Try SNR levels such as:

```text
20 dB
10 dB
5 dB
0 dB
```

Then report performance under noise.

This is a strong research component for the repository.

---

# 43. Robustness Benchmark

Create a benchmark specifically for degradation.

Example:

| Condition | Macro F1 |
|---|---:|
| Clean | — |
| +20 dB noise | — |
| +10 dB noise | — |
| +5 dB noise | — |
| MP3 compression | — |
| Reverb | — |
| Laptop microphone | — |

Again, values must come from real experiments.

This benchmark can become one of the most interesting parts of the project.

---

# 44. Custom Dataset

Eventually create:

```text
HearSafe-Rooms
```

A tiny internal test dataset recorded with:

```text
laptop microphone
phone microphone
headset microphone
large room
small room
background fan
TV noise
music
```

Do not mix those recordings into the test set accidentally after training on them.

Keep a truly unseen test set.

---

# 45. Advanced Feature — Sound Direction

Version 2 could support stereo or multiple microphones.

Goal:

```text
← SOUND               SOUND →
```

Potential approach:

```text
left microphone arrival time
        vs
right microphone arrival time
```

Estimate:

```text
Time Difference of Arrival
TDOA
```

Then UI could show:

```text
SIREN

<--------- ●

Likely direction: LEFT
```

Do **not** include this in the first MVP.

---

# 46. Advanced Feature — Event History

Store:

```text
timestamp
label
confidence
duration
```

Example:

```json
{
  "timestamp": "2026-09-29T21:32:04",
  "label": "door_knock",
  "confidence": 0.938,
  "duration_ms": 900
}
```

Could later generate:

```text
Today's detected events

Door knock     4
Dog bark       9
Car horn       3
Alarm          0
```

---

# 47. Advanced Feature — Personalized Classes

Long-term:

```text
"Learn my doorbell"
```

User records:

```text
10–20 examples
```

HearSafe creates a personal detector.

Potential architecture:

```text
pretrained audio encoder
       ↓
audio embeddings
       ↓
few-shot classifier
```

This is a very interesting future research direction.

---

# 48. Advanced Feature — Zero-Shot Sounds

Models such as CLAP can compare:

```text
audio embedding
```

with:

```text
text embedding
```

Potential future input:

```text
"smoke alarm"
"microwave beep"
"doorbell"
"cat meowing"
```

without manually training a new classification head for every label.

This should be experimental, not the initial core.

---

# 49. Testing Strategy

Unit tests:

```text
test_resample()
test_mono_conversion()
test_padding()
test_spectrogram_shape()
test_dataset_labels()
test_model_output_shape()
test_prediction_probabilities()
test_smoothing()
test_cooldown()
```

Integration tests:

```text
.wav file
   ↓
full pipeline
   ↓
prediction object
```

Performance test:

```text
100 inference runs
   ↓
mean latency
p95 latency
```

---

# 50. GitHub Actions

Workflow:

```text
push / pull request
       ↓
install dependencies
       ↓
lint
       ↓
unit tests
       ↓
small inference test
```

Possible tools:

```text
ruff
pytest
mypy
```

Do not train the full model in CI.

---

# 51. README Strategy

The repository README should immediately answer:

1. What does this project do?
2. Why does it exist?
3. Can I see it working?
4. How do I install it?
5. How well does it perform?
6. How was it trained?

Recommended README structure:

```text
# HearSafe

One-sentence description

[demo GIF]

## Why HearSafe?

## Features

## Demo

## Installation

## Quick Start

## Supported Sounds

## Model Architecture

## Datasets

## Training

## Benchmarks

## Real-Time Inference

## Privacy

## Limitations

## Roadmap

## Contributing

## License

## Citation
```

---

# 52. README Opening Example

```markdown
# HearSafe

HearSafe is an open-source, local-first environmental sound recognition
system that converts important sounds such as alarms, sirens, door knocks,
crying, and glass breaking into real-time visual alerts.

The project is designed as an assistive sound-awareness tool for deaf and
hard-of-hearing users while keeping microphone inference on-device.
```

---

# 53. Demo Is Extremely Important

A project people can see immediately is easier to understand.

Create a short GIF:

```text
microphone listening

      ↓

person knocks on door

      ↓

DOOR KNOCK — 94%
```

The GIF belongs near the top of README.

A visitor should understand the project in approximately 10 seconds.

---

# 54. Don't Commit Giant Datasets

`.gitignore`:

```gitignore
data/
datasets/
checkpoints/
runs/
wandb/
__pycache__/
*.pt
*.pth
*.onnx
```

For model releases, use:

```text
GitHub Releases
or
Hugging Face
```

instead of Git history.

---

# 55. Model Card

Create:

```text
MODEL_CARD.md
```

Include:

```text
Model description
Intended use
Training dataset
Input format
Output classes
Evaluation metrics
Known limitations
Ethical/safety considerations
License
```

This makes the repository look significantly more professional.

---

# 56. Dataset Documentation

Create:

```text
docs/datasets.md
```

For every dataset record:

```text
name
official URL
citation
classes used
license
download method
preprocessing
split
```

Example:

```text
ESC-50

Used classes:
- crying_baby
- footsteps
- coughing
- door_knock
- clock_alarm
- glass_breaking
- siren
- car_horn
- dog
- crackling_fire

Original sample rate:
44.1 kHz

HearSafe sample rate:
16 kHz
```

---

# 57. Suggested First Label Mapping

ESC-50 names should be converted into consistent internal names.

Example:

```python
LABELS = {
    "crying_baby": "baby_cry",
    "footsteps": "footsteps",
    "coughing": "cough",
    "door_wood_knock": "door_knock",
    "clock_alarm": "alarm",
    "glass_breaking": "glass_breaking",
    "siren": "siren",
    "car_horn": "car_horn",
    "dog": "dog_bark",
    "crackling_fire": "fire"
}
```

The exact ESC-50 category strings should be checked from `meta/esc50.csv` when implementing.

---

# 58. Class Imbalance

Later datasets will not be perfectly balanced.

Do not blindly optimize accuracy.

Possible techniques:

```text
weighted loss
weighted sampler
balanced mini-batches
focal loss
data augmentation
```

Always compare before/after experimentally.

---

# 59. Single-Label → Multi-Label Roadmap

Version 0.1:

```text
single-label
softmax
cross-entropy
```

Later:

```text
multi-label
sigmoid
binary cross-entropy
```

Single-label:

```text
P(siren) + P(glass) + P(door) = 1
```

Multi-label:

```text
P(siren) = 0.92
P(traffic) = 0.81
P(speech) = 0.66
```

No requirement for probabilities to sum to one.

Real life is naturally multi-label.

---

# 60. Potential Research Questions

Once the product works, HearSafe can become a small research platform.

Interesting questions:

### Q1

How well does a CNN trained on ESC-50 generalize to microphone recordings?

### Q2

How much does noise augmentation improve real-world performance?

### Q3

CNN from scratch vs pretrained embeddings?

### Q4

16 kHz vs 32 kHz?

### Q5

64 vs 128 mel bands?

### Q6

1-second vs 5-second inference windows?

### Q7

How much accuracy is lost when exporting to ONNX / quantizing?

### Q8

How does model performance change under reverberation?

### Q9

Does SpecAugment improve low-data environmental classification?

### Q10

How much latency can be reduced without losing event recall?

These can become GitHub issues or experiment reports.

---

# 61. Suggested GitHub Issues

Immediately create project issues such as:

```text
[DATA] Add ESC-50 loader
[AUDIO] Implement log-mel preprocessing
[MODEL] Create baseline CNN
[TRAIN] Implement training loop
[EVAL] Generate confusion matrix
[INFERENCE] Add WAV inference
[STREAM] Add microphone ring buffer
[ALERT] Implement confidence smoothing
[UI] Create event dashboard
[EXPORT] ONNX model export
[DOCS] Add model card
[TEST] Audio pipeline tests
```

This makes development structured and shows repository maturity.

---

# 62. Branch Strategy

Simple:

```text
main
develop
feature/esc50-loader
feature/cnn-model
feature/realtime-audio
feature/ui
```

For a one-person project, do not over-engineer Git workflow.

---

# 63. Semantic Versions

Suggested versions:

```text
v0.1.0
offline WAV classifier

v0.2.0
real-time microphone detection

v0.3.0
desktop/web notifications

v0.4.0
FSD50K expanded training

v0.5.0
multi-label recognition

v1.0.0
stable local application
```

---

# 64. Suggested Roadmap

## v0.1 — Research Prototype

```text
ESC-50
log-mel pipeline
CNN
training
evaluation
WAV inference
```

## v0.2 — Live Detection

```text
microphone input
sliding windows
temporal smoothing
cooldowns
```

## v0.3 — Application

```text
GUI
event history
settings
notifications
```

## v0.4 — Better Data

```text
FSD50K subset
background class
noise augmentation
cross-dataset testing
```

## v0.5 — Deployment

```text
ONNX
CPU optimizations
quantization
installer
```

## v1.0

```text
stable API
tests
documentation
model card
benchmarks
release package
```

---

# 65. What Makes This Repository Interesting?

Do not market HearSafe as:

> "CNN trained on ESC-50."

That already exists thousands of times.

Market it as:

> **A local-first real-time environmental sound awareness engine.**

The interesting part is the system around the classifier.

Specifically:

```text
training
+
real-time streaming
+
temporal detection
+
visual alerts
+
privacy
+
robustness benchmarking
+
deployment
```

That combination makes the project much stronger.

---

# 66. Potential Features That Can Attract Contributors

Good open-source extension points:

```text
custom sound classes
new datasets
new models
mobile application
Raspberry Pi support
smartwatch notifications
direction detection
Home Assistant integration
MQTT
Discord integration
WebSocket event stream
custom alert rules
```

Example:

```text
HearSafe detects door knock
        ↓
MQTT
        ↓
smart light flashes
```

That gives the project a natural ecosystem.

---

# 67. GitHub Topics

Suggested repository topics:

```text
machine-learning
deep-learning
audio
audio-classification
pytorch
accessibility
assistive-technology
sound-event-detection
environmental-sound-classification
signal-processing
real-time
computer-audition
```

---

# 68. Repository Description

Suggested:

```text
Local-first AI for real-time environmental sound recognition and visual alerts.
```

Alternative:

```text
Open-source environmental sound recognition for real-time accessibility alerts.
```

---

# 69. Possible Tagline

```text
See the sounds around you.
```

Keep the technical README professional even if a short tagline is used.

---

# 70. Biggest Technical Risks

## Risk 1 — Small dataset

ESC-50 is too small for a robust final product.

Solution:

```text
ESC-50 for development
FSD50K for expansion
custom recordings for validation
```

---

## Risk 2 — Dataset/domain mismatch

Internet field recordings differ from:

```text
laptop microphones
phone microphones
rooms
apartments
offices
```

Solution:

```text
noise augmentation
reverb augmentation
cross-dataset tests
custom test recordings
```

---

## Risk 3 — False Alerts

Solution:

```text
confidence threshold
temporal smoothing
cooldown
unknown/background detection
```

---

## Risk 4 — Missed Sounds

Solution:

Measure:

```text
recall per important class
```

not just global accuracy.

---

## Risk 5 — Model Too Heavy

Solution:

```text
small CNN
MobileNet
ONNX
quantization
```

---

# 71. Recommended First Technical Stack

If starting today:

```text
Language:
Python 3.11+

ML:
PyTorch
torchaudio

Audio:
librosa
sounddevice

Data:
pandas
numpy

Metrics:
scikit-learn

API:
FastAPI

Visualization:
matplotlib

Testing:
pytest

Lint:
ruff
```

---

# 72. First Week Checklist

## Day 1

```text
[ ] Create GitHub repository
[ ] Add README skeleton
[ ] Add Python project structure
[ ] Add .gitignore
[ ] Add license
[ ] Download ESC-50
```

## Day 2

```text
[ ] Inspect metadata
[ ] Select classes
[ ] Plot waveform
[ ] Plot mel spectrogram
[ ] Implement audio loader
```

## Day 3

```text
[ ] Implement Dataset class
[ ] Implement preprocessing
[ ] Write tests
```

## Day 4

```text
[ ] Create baseline CNN
[ ] Run first training
[ ] Save checkpoint
```

## Day 5

```text
[ ] Evaluate
[ ] confusion matrix
[ ] per-class metrics
[ ] fix obvious problems
```

## Day 6

```text
[ ] WAV inference CLI
[ ] clean package structure
```

## Day 7

```text
[ ] microphone prototype
[ ] first README demo
[ ] publish initial release
```

This is a development sequence, not a deadline.

---

# 73. First Implementation Target

The first meaningful milestone should be:

```bash
python scripts/train.py
```

followed by:

```bash
python scripts/predict.py \
    --audio samples/door_knock.wav
```

returning:

```text
door_knock 0.94
```

Do not start by building the frontend.

First prove:

```text
DATA → MODEL → PREDICTION
```

Then build real-time infrastructure.

---

# 74. First Research Benchmark

Run:

```text
Model 1:
MFCC + SVM

Model 2:
Log-Mel + CNN

Model 3:
Pretrained embedding + MLP
```

Then compare:

```text
accuracy
macro F1
model size
training time
CPU latency
```

That table should become one of the main README elements.

---

# 75. Suggested Architecture Diagram for README

```text
                       HearSafe
                          │
                          ▼
                    Audio Source
                ┌─────────┴─────────┐
                │                   │
             WAV File           Microphone
                │                   │
                └─────────┬─────────┘
                          ▼
                 Audio Preprocessor
                          │
                          ▼
                Log-Mel Spectrogram
                          │
                          ▼
                 Classification Model
                          │
                          ▼
                  Event Probabilities
                          │
                          ▼
                  Temporal Smoothing
                          │
                          ▼
                   Alert Manager
                    ┌─────┴─────┐
                    ▼           ▼
               Desktop UI      API
```

---

# 76. Long-Term Architecture

```text
                    ┌────────────────────┐
                    │     Microphone     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Streaming Buffer   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Audio Frontend     │
                    │                    │
                    │ Resample           │
                    │ Normalize          │
                    │ Log-Mel            │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Neural Network     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Event Processor    │
                    │                    │
                    │ Smooth             │
                    │ Threshold          │
                    │ Cooldown           │
                    └─────────┬──────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
           Desktop UI      REST API       WebSocket
```

---

# 77. Definition of Done for v0.1

Version `v0.1.0` is done when:

```text
[ ] repository installs from fresh environment
[ ] ESC-50 data preparation documented
[ ] selected classes train correctly
[ ] model checkpoint can be saved
[ ] test-set metrics generated automatically
[ ] confusion matrix generated
[ ] CLI accepts WAV files
[ ] predictions include confidence
[ ] unit tests pass
[ ] README contains architecture
[ ] README contains real benchmark numbers
[ ] dataset licenses/citations documented
[ ] MODEL_CARD.md exists
```

---

# 78. Definition of Done for v0.2

```text
[ ] microphone streaming works
[ ] inference does not block audio capture
[ ] sliding windows implemented
[ ] prediction smoothing implemented
[ ] cooldown implemented
[ ] notification generated
[ ] configurable threshold
[ ] CPU latency benchmarked
```

---

# 79. Recommended Scope for the First GitHub Release

Do not wait until everything is complete.

A strong first release could simply be:

```text
HearSafe v0.1

✓ ESC-50 pipeline
✓ 10 accessibility-relevant sound classes
✓ CNN training
✓ evaluation
✓ CLI inference
✓ spectrogram visualization
✓ reproducible configs
✓ benchmark report
```

Then publish a roadmap showing:

```text
Real-time microphone mode → next
```

This gives people something functional while development continues.

---

# 80. Source List

## ESC-50

Repository:

https://github.com/karolpiczak/ESC-50

Paper:

K. J. Piczak,  
"ESC: Dataset for Environmental Sound Classification",  
ACM Multimedia, 2015.

---

## UrbanSound8K

Dataset information:

https://urbansounddataset.weebly.com/urbansound8k.html

Freesound dataset catalog:

https://labs.freesound.org/datasets/

Paper:

J. Salamon, C. Jacoby, J. P. Bello,  
"A Dataset and Taxonomy for Urban Sound Research",  
ACM Multimedia, 2014.

---

## FSD50K

Official explorer:

https://fsannotator.upf.edu/fsd/release/FSD50K/

Zenodo:

https://zenodo.org/records/4060432

Paper:

E. Fonseca, X. Favory, J. Pons, F. Font, X. Serra,  
"FSD50K: An Open Dataset of Human-Labeled Sound Events",  
IEEE/ACM TASLP, 2022.

---

## AudioSet

Official:

https://research.google.com/audioset/

Dataset:

https://research.google.com/audioset/dataset/index.html

---

## YAMNet

TensorFlow tutorial:

https://www.tensorflow.org/hub/tutorials/yamnet

Source:

https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

---

# 81. Final Project Direction

The recommended development path is:

```text
STEP 1
ESC-50 subset
       ↓
STEP 2
MFCC + SVM baseline
       ↓
STEP 3
Log-Mel CNN
       ↓
STEP 4
Evaluation + benchmarks
       ↓
STEP 5
WAV inference CLI
       ↓
STEP 6
Real-time microphone stream
       ↓
STEP 7
Temporal event detector
       ↓
STEP 8
Visual notifications
       ↓
STEP 9
FSD50K expansion
       ↓
STEP 10
ONNX / desktop deployment
```

The important principle is:

> **HearSafe should be built as an actual audio event detection product, not merely an ML notebook.**

The dataset and neural network are the core intelligence.

The real value of the repository will come from combining that intelligence with:

```text
good engineering
+
real-time inference
+
accessibility
+
privacy
+
reproducibility
+
honest benchmarks
```

That combination is what can make HearSafe a credible open-source AI project rather than another environmental-sound-classification tutorial.

---

# 82. Immediate Next Task

The first development task should be:

```text
1. Create the repository structure.
2. Download ESC-50.
3. Read `meta/esc50.csv`.
4. Extract the selected HearSafe classes.
5. Generate a class-distribution report.
6. Plot one waveform and one log-mel spectrogram per class.
7. Build the PyTorch Dataset/DataLoader.
```

Only after these steps should the first neural network be written.

That gives us a clean starting point for the entire project.
