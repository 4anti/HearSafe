"""Verified model packages with lazy CPU runtimes; never load pickled weights."""

import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .audio import log_mel
from .types import Label, ModelManifest

YAMNET_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
YAMNET_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"
LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
LABELS_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"


def default_model_dir() -> Path:
    if os.environ.get("HEARSAFE_MODEL_DIR"):
        return Path(os.environ["HEARSAFE_MODEL_DIR"])
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "models" / "yamnet"
    candidates = [
        Path(__file__).resolve().parents[2] / "models" / "yamnet",
        Path.cwd() / "models" / "yamnet",
        Path.home() / ".hearsafe" / "models" / "yamnet",
    ]
    return next((path for path in candidates if (path / "manifest.json").is_file()), candidates[-1])


def download_yamnet(destination: Path | str | None = None) -> Path:
    """Explicit network operation. Detection itself never downloads anything."""
    destination = Path(destination) if destination else default_model_dir()
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(YAMNET_URL, timeout=60) as response:
        binary = response.read()
    if hashlib.sha256(binary).hexdigest() != YAMNET_SHA256:
        raise ValueError("YAMNet download checksum does not match the pinned model")
    with urllib.request.urlopen(LABELS_URL, timeout=30) as response:
        label_bytes = response.read()
    if hashlib.sha256(label_bytes).hexdigest() != LABELS_SHA256:
        raise ValueError("YAMNet label download does not match the pinned class map")
    rows = list(csv.DictReader(io.StringIO(label_bytes.decode("utf-8"))))
    if len(rows) != 521 or [int(row["index"]) for row in rows] != list(range(521)):
        raise ValueError("YAMNet label map is incompatible with the pinned model")
    labels = [{"id": row["mid"], "name": row["display_name"]} for row in rows]
    existing = destination / "manifest.json"
    if existing.is_file():
        old = json.loads(existing.read_text(encoding="utf-8"))
        pinned_labels = old.get("provenance", {}).get("labels_sha256")
        if pinned_labels and hashlib.sha256(label_bytes).hexdigest() != pinned_labels:
            raise ValueError("YAMNet label map changed; review it before updating the pin")
    (destination / "model.tflite").write_bytes(binary)
    (destination / "labels.csv").write_bytes(label_bytes)
    manifest = {
        "schema_version": 1,
        "model_id": "yamnet-mediapipe-v1",
        "name": "YAMNet sound explorer (521 categories)",
        "backend": "litert",
        "sample_rate": 16000,
        "window_samples": 15600,
        "hop_samples": 7680,
        "model_file": "model.tflite",
        "sha256": YAMNET_SHA256,
        "labels": labels,
        "frontend": "waveform",
        "license": "Apache-2.0",
        "provenance": {
            "model_url": YAMNET_URL,
            "model_card": "https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1",
            "labels_url": LABELS_URL,
            "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with urllib.request.urlopen(
        "https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=30
    ) as response:
        (destination / "LICENSE.txt").write_bytes(response.read())
    (destination / "NOTICE.txt").write_text(
        "YAMNet pretrained audio classifier by Google, distributed under Apache 2.0.\n"
        "Source/model provenance is recorded in manifest.json. HearSafe code is MIT.\n"
        "This model is included unmodified. Model scores are not calibrated correctness probabilities.\n",
        encoding="utf-8",
    )
    return destination


def read_manifest(path: Path | str) -> tuple[ModelManifest, Path]:
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Model manifest not found in {directory}. Run hearsafe download-model.")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["labels"] = [Label(**label) for label in raw["labels"]]
        # Allow extra explanatory metadata without changing the public manifest fields.
        fields = ModelManifest.__dataclass_fields__
        manifest = ModelManifest(**{key: value for key, value in raw.items() if key in fields})
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid model manifest: {exc}") from exc
    if manifest.schema_version != 1 or manifest.backend not in {"litert", "onnx"}:
        raise ValueError("Unsupported model schema or runtime backend")
    sizes = (manifest.sample_rate, manifest.window_samples, manifest.hop_samples)
    if any(not isinstance(size, int) or size <= 0 for size in sizes):
        raise ValueError("Model sample rate and window sizes must be positive integers")
    if manifest.hop_samples > manifest.window_samples:
        raise ValueError("Model hop cannot exceed its analysis window")
    if any(
        not isinstance(label.id, str)
        or not label.id.strip()
        or not isinstance(label.name, str)
        or not label.name.strip()
        for label in manifest.labels
    ):
        raise ValueError("Model label identifiers and names must be nonempty strings")
    if not manifest.labels or len({label.id for label in manifest.labels}) != len(manifest.labels):
        raise ValueError("Model labels must be nonempty and have unique identifiers")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (manifest.model_id, manifest.name, manifest.model_file)
    ):
        raise ValueError("Model identifier, name, and file must be nonempty strings")
    binary_path = (directory / manifest.model_file).resolve()
    if not binary_path.is_relative_to(directory.resolve()):
        raise ValueError("Model file must be inside the model package")
    if not binary_path.is_file():
        raise ValueError(f"Model file missing: {binary_path.name}")
    if hashlib.sha256(binary_path.read_bytes()).hexdigest() != manifest.sha256:
        raise ValueError("Model checksum mismatch; replace the damaged model package")
    return manifest, binary_path


class RuntimeModel:
    def __init__(self, directory: Path | str):
        self.manifest, path = read_manifest(directory)
        if self.manifest.backend == "litert":
            from ai_edge_litert.interpreter import Interpreter

            self._runtime = Interpreter(model_path=str(path), num_threads=2)
            self._runtime.allocate_tensors()
            self._input = self._runtime.get_input_details()[0]
            self._output = self._runtime.get_output_details()[0]
            shape = list(self._input["shape"])
            if self._input["dtype"] != np.float32 or shape not in (
                [self.manifest.window_samples],
                [1, self.manifest.window_samples],
            ):
                raise ValueError("LiteRT model input does not match the manifest")
            if int(np.prod(self._output["shape"])) != len(self.manifest.labels):
                raise ValueError("LiteRT model output does not match its labels")
        else:
            import onnxruntime as ort

            if (
                self.manifest.frontend != "esc50-log-mel-v1"
                or self.manifest.window_samples != 80000
                or self.manifest.sample_rate != 16000
            ):
                raise ValueError("Unsupported ONNX audio frontend")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            # Audio windows arrive every 480 ms: sleep between jobs to save CPU/power.
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self._runtime = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._input = self._runtime.get_inputs()[0]
            self._output = self._runtime.get_outputs()[0]
            if (
                self._input.type != "tensor(float)"
                or self._output.type != "tensor(float)"
                or self._input.shape != [1, 1, 64, 498]
                or self._output.shape != [1, len(self.manifest.labels)]
            ):
                raise ValueError("ONNX model tensors do not match the frontend or labels")

    def predict(self, window: np.ndarray) -> np.ndarray:
        try:
            return self._predict(window)
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc

    def _predict(self, window: np.ndarray) -> np.ndarray:
        if self.manifest.backend == "litert":
            self._runtime.set_tensor(
                self._input["index"], window.reshape(self._input["shape"]).astype(np.float32)
            )
            self._runtime.invoke()
            return self._runtime.get_tensor(self._output["index"]).reshape(-1).copy()
        features = log_mel(window)[None, None]
        logits = self._runtime.run([self._output.name], {self._input.name: features})[0].reshape(-1)
        values = np.exp(logits - np.max(logits))
        return (values / values.sum()).astype(np.float32)


def load_model(path: Path | str | None = None) -> RuntimeModel:
    try:
        return RuntimeModel(path or default_model_dir())
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Model runtime could not start: {exc}") from exc
