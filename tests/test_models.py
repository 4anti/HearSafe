import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hearsafe.models import load_model, read_manifest


def manifest_fixture(tmp_path):
    binary = b"test model"
    (tmp_path / "model.tflite").write_bytes(binary)
    manifest = {
        "schema_version": 1,
        "model_id": "test",
        "name": "Test",
        "backend": "litert",
        "sample_rate": 16000,
        "window_samples": 15600,
        "hop_samples": 7680,
        "labels": [{"id": "a", "name": "A"}],
        "model_file": "model.tflite",
        "sha256": hashlib.sha256(binary).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_checksum_failure_before_runtime_loading(tmp_path):
    manifest_fixture(tmp_path)
    (tmp_path / "model.tflite").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model_file": "../elsewhere.tflite"}, "inside"),
        ({"schema_version": 2}, "schema"),
        ({"hop_samples": 20000}, "hop"),
        ({"labels": [{"id": "a", "name": "A"}, {"id": "a", "name": "B"}]}, "unique"),
        ({"labels": [{"id": 1, "name": "A"}]}, "strings"),
        ({"labels": [{"id": "a", "name": ""}]}, "strings"),
    ],
)
def test_incompatible_model_manifest(tmp_path, change, message):
    manifest = manifest_fixture(tmp_path)
    manifest.update(change)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        read_manifest(tmp_path)


def test_backend_construction_error_is_readable(tmp_path):
    manifest = manifest_fixture(tmp_path)
    manifest.update(backend="onnx", frontend="esc50-log-mel-v1", window_samples=80000)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Model runtime could not start"):
        load_model(tmp_path)


def test_onnx_frontend_rejects_wrong_sample_rate(tmp_path):
    manifest = manifest_fixture(tmp_path)
    manifest.update(
        backend="onnx", frontend="esc50-log-mel-v1", window_samples=80000, sample_rate=48000
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frontend"):
        load_model(tmp_path)


def test_pinned_yamnet_runs_without_tensorflow():
    path = Path("models/yamnet")
    if not (path / "model.tflite").is_file():
        pytest.skip("Pinned model is downloaded in the portable CI job")
    import sys

    model = load_model(path)
    scores = model.predict(np.zeros(15600, np.float32))
    assert scores.shape == (521,) and np.isfinite(scores).all()
    assert all(0 <= value <= 1 for value in scores)
    # Training tests may already have imported torch; the runtime must not add TensorFlow.
    assert "tensorflow" not in sys.modules
