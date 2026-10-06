"""Release checks independent of a Windows compiler or downloaded model."""

import importlib.util
import zipfile
from pathlib import Path

import pytest


def build_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_portable.py"
    spec = importlib.util.spec_from_file_location("hearsafe_build_portable", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zip_is_stable_and_retains_portable_folder(tmp_path):
    bundle = tmp_path / "HearSafe"
    (bundle / "models" / "yamnet").mkdir(parents=True)
    (bundle / "hearsafe-cli.exe").write_bytes(b"fake executable")
    (bundle / "models" / "yamnet" / "manifest.json").write_text("{}")
    module = build_module()
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    module.deterministic_zip(bundle, first)
    # File-system metadata must not affect the ZIP.
    (bundle / "hearsafe-cli.exe").touch()
    module.deterministic_zip(bundle, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "HearSafe/hearsafe-cli.exe",
            "HearSafe/models/yamnet/manifest.json",
        ]


def test_public_bundle_rejects_research_weights(tmp_path):
    (tmp_path / "model.tflite").write_bytes(b"model")
    (tmp_path / "model.onnx").write_bytes(b"research weights")
    with pytest.raises(ValueError, match="Research weights"):
        build_module().validate_model(tmp_path)


def test_bundle_requires_model_and_notices(tmp_path):
    module = build_module()
    with pytest.raises(ValueError, match="model.tflite"):
        module.validate_model(tmp_path)
    (tmp_path / "model.tflite").write_bytes(b"model")
    with pytest.raises(ValueError, match="manifest"):
        module.validate_model(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="license"):
        module.validate_model(tmp_path)
