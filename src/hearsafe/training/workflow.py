"""Reproducible local ESC-50 fold training, evaluation, and ONNX export."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import classification_metrics

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 80_000
FEATURE_SHAPE = (64, 498)
FRONTEND_ID = "esc50-log-mel-v1"


def read_metadata(dataset: Path, expected_classes: int | None = 50):
    """Return metadata in CSV order with a canonical target-index label list."""
    dataset = Path(dataset)
    source = dataset / "meta" / "esc50.csv"
    if not source.is_file():
        raise FileNotFoundError(f"ESC-50 metadata was not found at {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("ESC-50 metadata is empty.")
    categories: dict[int, str] = {}
    seen_filenames: set[str] = set()
    for row in rows:
        try:
            row["fold"] = int(row["fold"])
            row["target"] = int(row["target"])
            filename = row["filename"]
            category = row["category"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ESC-50 metadata has missing or invalid columns.") from exc
        if row["fold"] not in (1, 2, 3, 4, 5) or row["target"] < 0:
            raise ValueError("ESC-50 folds must be 1–5 and targets nonnegative.")
        if Path(filename).name != filename or filename in seen_filenames:
            raise ValueError(f"Invalid or duplicate audio filename: {filename}")
        seen_filenames.add(filename)
        if not (dataset / "audio" / filename).is_file():
            raise FileNotFoundError(f"ESC-50 audio was not found: {filename}")
        target = row["target"]
        if target in categories and categories[target] != category:
            raise ValueError(f"Target {target} has inconsistent category names.")
        categories[target] = category
    if sorted(categories) != list(range(len(categories))):
        raise ValueError("Class targets must be contiguous and start at zero.")
    if expected_classes is not None and len(categories) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes; found {len(categories)}.")
    return rows, [categories[index] for index in range(len(categories))]


def split_indices(rows):
    """Preserve the dataset's official folds; never use a random split."""
    result = {
        "train": np.asarray([i for i, row in enumerate(rows) if row["fold"] <= 3]),
        "validation": np.asarray([i for i, row in enumerate(rows) if row["fold"] == 4]),
        "test": np.asarray([i for i, row in enumerate(rows) if row["fold"] == 5]),
    }
    if any(not indices.size for indices in result.values()):
        raise ValueError("Folds 1–3, 4, and 5 must all contain audio clips.")
    return result


def _cross_fold_sources(rows) -> dict[str, list[int]]:
    sources: dict[str, set[int]] = {}
    for row in rows:
        if row.get("src_file"):
            sources.setdefault(row["src_file"], set()).add(row["fold"])
    return {source: sorted(folds) for source, folds in sources.items() if len(folds) > 1}


def normalization(features: np.ndarray, train_indices: np.ndarray):
    training = np.asarray(features[train_indices], dtype=np.float32)
    mean = training.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = training.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def _prepare_audio(path: Path) -> np.ndarray:
    from hearsafe.audio import load_wav, resample_audio

    audio, sample_rate = load_wav(path)
    audio = resample_audio(audio, sample_rate, SAMPLE_RATE)
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    return np.asarray(audio[:WINDOW_SAMPLES], dtype=np.float32)


def _feature_fingerprint(dataset: Path, rows) -> str:
    fingerprint = hashlib.sha256((dataset / "meta" / "esc50.csv").read_bytes())
    fingerprint.update(FRONTEND_ID.encode())
    # Include the actual frontend so a numerical implementation change invalidates cache.
    import hearsafe.audio as audio_module

    fingerprint.update(Path(audio_module.__file__).read_bytes())
    for row in rows:
        stat = (dataset / "audio" / row["filename"]).stat()
        fingerprint.update(f"{row['filename']}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return fingerprint.hexdigest()


def _features(dataset: Path, rows, output: Path) -> np.ndarray:
    from hearsafe.audio import log_mel

    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    feature_file = cache / "features.npy"
    cache_manifest = cache / "features.json"
    fingerprint = _feature_fingerprint(dataset, rows)
    if feature_file.is_file() and cache_manifest.is_file():
        try:
            cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                features = np.load(feature_file, mmap_mode="r")
                if features.shape == (len(rows), *FEATURE_SHAPE) and features.dtype == np.float32:
                    return features
                del features
        except (OSError, ValueError):
            pass  # Rebuild a damaged cache from source audio.
    # An interrupted preparation must never leave a cache marked as complete.
    cache_manifest.unlink(missing_ok=True)
    features = np.lib.format.open_memmap(
        feature_file, mode="w+", dtype=np.float32, shape=(len(rows), *FEATURE_SHAPE)
    )
    for index, row in enumerate(rows):
        feature = log_mel(_prepare_audio(dataset / "audio" / row["filename"]))
        if feature.shape != FEATURE_SHAPE or not np.isfinite(feature).all():
            raise ValueError(f"Invalid features for {row['filename']}: {feature.shape}")
        features[index] = feature
        if (index + 1) % 200 == 0 or index + 1 == len(rows):
            print(f"Prepared {index + 1}/{len(rows)} audio clips", file=sys.stderr, flush=True)
    features.flush()
    cache_manifest.write_text(
        json.dumps({"fingerprint": fingerprint, "frontend": FRONTEND_ID}, indent=2) + "\n",
        encoding="utf-8",
    )
    return np.load(feature_file, mmap_mode="r")


def _torch_predictions(model, loader):
    import torch

    targets, predictions, losses = [], [], []
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    with torch.inference_mode():
        for features, expected in loader:
            logits = model(features)
            losses.append(float(criterion(logits, expected)))
            targets.extend(expected.numpy().tolist())
            predictions.extend(logits.argmax(dim=1).numpy().tolist())
    return targets, predictions, sum(losses) / len(targets)


def _export(model, sample, output: Path, labels: list[str]) -> dict:
    import onnxruntime as ort
    import torch

    model.eval()
    model_file = output / "model.onnx"
    torch.onnx.export(
        model,
        sample,
        str(model_file),
        input_names=["log_mel"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
    )
    with torch.inference_mode():
        expected = model(sample).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
    error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(actual, expected, atol=1e-4, rtol=1e-4):
        raise RuntimeError(f"ONNX export failed PyTorch parity: max logit error {error}")
    manifest = {
        "schema_version": 1,
        "model_id": "esc50-cnn-v1",
        "name": "HearSafe ESC-50 research",
        "backend": "onnx",
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": 7680,
        "model_file": "model.onnx",
        "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "labels": [{"id": f"esc50:{i}", "name": name} for i, name in enumerate(labels)],
        "frontend": FRONTEND_ID,
        "license": "Research; ESC-50 dataset CC BY-NC 3.0",
        "provenance": {
            "dataset": "ESC-50",
            "dataset_url": "https://github.com/karolpiczak/ESC-50",
            "train_folds": [1, 2, 3],
            "validation_fold": 4,
            "held_out_fold": 5,
            "weights_distribution": "Local research output; not included in the public portable download.",
        },
        "preprocessing": {
            "mel_bands": 64,
            "mel_scale": "HTK",
            "mel_min_hz": 125,
            "mel_max_hz": 7500,
            "spectrum": "magnitude",
            "log_offset": 0.001,
            "center": False,
            "window": "Hann",
            "window_samples": 400,
            "hop_samples": 160,
            "fft_samples": 512,
            "feature_shape": [1, 1, 64, 498],
            "normalization": "Per-mel training-fold mean and standard deviation embedded in ONNX graph",
        },
        "export": {
            "opset": 18,
            "exporter": "torch.onnx legacy (dynamo=False)",
            "max_logit_error": error,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_report(report: dict, output: Path, stem: str):
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics = report.get("test", report)
    labels = report["labels"]
    lines = [
        f"# HearSafe ESC-50 evaluation: fold {metrics.get('fold', 5)}",
        "",
        "These results measure prerecorded ESC-50 clips. They do not establish microphone or emergency-alert reliability.",
        "",
        f"Training folds: 1–3. Checkpoint selection: fold 4. Evaluation: fold {metrics.get('fold', 5)}.",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Model bytes: {report['model_size_bytes']:,}",
        f"- Mean inference time: {report['inference']['mean_ms']:.2f} ms",
        f"- p95 inference time: {report['inference']['p95_ms']:.2f} ms",
        "",
        "| Category | Test clips | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for i, name in enumerate(labels):
        lines.append(
            f"| {name} | {metrics['support'][i]} | {metrics['per_class_recall'][i]:.4f} | {metrics['per_class_f1'][i]:.4f} |"
        )
    if report.get("cross_fold_sources"):
        lines.extend(
            [
                "",
                f"The supplied metadata has {len(report['cross_fold_sources'])} source identifiers shared by official folds. The JSON report lists them; this experiment preserves the official folds.",
            ]
        )
    lines.extend(
        [
            "",
            "The JSON report includes the full confusion matrix (rows: actual labels; columns: predicted labels).",
            "",
        ]
    )
    (output / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(dataset: Path, modeldir: Path, fold: int = 5) -> dict[str, Any]:
    """Evaluate an exported model through the same detector used by the desktop app."""
    from hearsafe.detector import Detector
    from hearsafe.models import load_model

    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("Evaluation fold must be 1–5.")
    dataset, modeldir = Path(dataset), Path(modeldir)
    rows, labels = read_metadata(dataset)
    runtime = load_model(modeldir)
    if runtime.manifest.model_id != "esc50-cnn-v1":
        raise ValueError("ESC-50 evaluation requires a HearSafe ESC-50 research model.")
    actual_names = [label.name for label in runtime.manifest.labels]
    if actual_names != labels:
        raise ValueError("Model labels do not match ESC-50 target order.")
    targets, predictions, timings = [], [], []
    selected = [row for row in rows if row["fold"] == fold]
    if not selected:
        raise ValueError(f"Evaluation fold {fold} is empty.")
    # Warm the session before measuring inference, excluding audio loading/resampling.
    runtime.predict(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    for index, row in enumerate(selected):
        audio = _prepare_audio(dataset / "audio" / row["filename"])
        detector = Detector(runtime)
        started = time.perf_counter()
        frames = detector.process(audio, SAMPLE_RATE)
        timings.append((time.perf_counter() - started) * 1000)
        if len(frames) != 1:
            raise RuntimeError(
                "A five-second ESC-50 clip must produce one complete detector window."
            )
        scores = [prediction.score for prediction in frames[0].predictions]
        targets.append(row["target"])
        predictions.append(int(np.argmax(scores)))
        if (index + 1) % 100 == 0:
            print(
                f"Evaluated {index + 1}/{len(selected)} fold-{fold} clips",
                file=sys.stderr,
                flush=True,
            )
    report = {
        "experiment": "Single ESC-50 fold evaluation; not five-fold cross-validation",
        "fold": fold,
        "fold_role": "held-out test" if fold == 5 else "validation" if fold == 4 else "training",
        "cross_fold_sources": _cross_fold_sources(rows),
        "labels": labels,
        **classification_metrics(targets, predictions, len(labels)),
        "model_size_bytes": (modeldir / "model.onnx").stat().st_size,
        "inference": {
            "mean_ms": float(np.mean(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "maximum_ms": float(max(timings)),
            "real_time_hop_ms": 480,
            "keeps_up_at_p95": bool(np.percentile(timings, 95) < 480),
            "includes": "Shared frontend and inference; excludes file reading",
        },
    }
    _write_report(report, modeldir, f"evaluation-fold-{fold}")
    return report


def train(
    dataset: Path,
    output: Path,
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 32,
    seed: int = 42,
    threads: int = 4,
) -> dict[str, Any]:
    """Train on folds 1–3, select on fold 4, export, then evaluate once on fold 5."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Training needs the optional HearSafe training dependencies.") from exc
    from .cnn import ResearchCNN

    if epochs < 1 or epochs > 50 or patience < 1 or batch_size < 1 or threads < 1:
        raise ValueError("Use 1–50 epochs and positive patience, batch size, and thread count.")
    dataset, output = Path(dataset), Path(output)
    rows, labels = read_metadata(dataset)
    splits = split_indices(rows)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    features = _features(dataset, rows, output)
    targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
    mean, std = normalization(features, splits["train"])
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.array(features[splits["train"]])).unsqueeze(1),
            torch.from_numpy(targets[splits["train"]]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.array(features[splits["validation"]])).unsqueeze(1),
            torch.from_numpy(targets[splits["validation"]]),
        ),
        batch_size=batch_size,
        num_workers=0,
    )
    model = ResearchCNN(mean, std, len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch, stale = 0, 0
    history = []
    started = time.perf_counter()
    checkpoint = output / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, examples = 0.0, 0
        epoch_started = time.perf_counter()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_targets)
            examples += len(batch_targets)
        expected, predicted, validation_loss = _torch_predictions(model, validation_loader)
        validation_metrics = classification_metrics(expected, predicted, len(labels))
        entry = {
            "epoch": epoch,
            "training_loss": total_loss / examples,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(entry)
        print(
            f"Epoch {epoch}/{epochs}: loss {entry['training_loss']:.4f}, validation loss {validation_loss:.4f}, accuracy {validation_metrics['accuracy']:.4f}",
            file=sys.stderr,
            flush=True,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        (output / "training-history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if stale >= patience:
            print(
                f"Early stopping after {patience} epochs without validation improvement",
                file=sys.stderr,
                flush=True,
            )
            break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    sample = torch.from_numpy(
        np.array(features[splits["validation"][0] : splits["validation"][0] + 1])
    ).unsqueeze(1)
    manifest = _export(model, sample, output, labels)
    manifest["provenance"].update(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "normalization_folds": [1, 2, 3],
            "metadata_sha256": hashlib.sha256(
                (dataset / "meta" / "esc50.csv").read_bytes()
            ).hexdigest(),
        }
    )
    if (dataset / "LICENSE").is_file():
        shutil.copyfile(dataset / "LICENSE", output / "DATASET_LICENSE.txt")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    test = evaluate(dataset, output, fold=5)
    report = {
        "experiment": "Single held-out ESC-50 fold; not five-fold cross-validation",
        "labels": labels,
        "splits": {name: int(indices.size) for name, indices in splits.items()},
        "cross_fold_sources": _cross_fold_sources(rows),
        "configuration": {
            "epochs_limit": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "seed": seed,
            "threads": threads,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "test": test,
        "model_size_bytes": test["model_size_bytes"],
        "inference": test["inference"],
        "export": manifest["export"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
        "license": "Local research weights trained from the full CC BY-NC ESC-50 dataset; excluded from public portable downloads.",
    }
    _write_report(report, output, "training-report")
    return report
