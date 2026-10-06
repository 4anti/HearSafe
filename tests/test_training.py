import csv
import importlib.util

import numpy as np
import pytest

from hearsafe.training.metrics import classification_metrics
from hearsafe.training.workflow import normalization, read_metadata, split_indices


def test_metrics_capture_missed_categories_and_confusion():
    metrics = classification_metrics([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["per_class_recall"] == pytest.approx([0.5, 1.0, 0.0])
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 0) / 3)


def test_normalization_cannot_use_validation_or_test_samples():
    features = np.ones((5, 64, 498), dtype=np.float32)
    features[0] = 2
    features[1] = 4
    features[3:] = 10_000
    mean, std = normalization(features, np.asarray([0, 1]))
    np.testing.assert_allclose(mean, 3)
    np.testing.assert_allclose(std, 1)


def test_split_preserves_official_folds():
    rows = [{"fold": fold, "src_file": str(fold)} for fold in (1, 2, 3, 4, 5)]
    splits = split_indices(rows)
    assert splits["train"].tolist() == [0, 1, 2]
    assert splits["validation"].tolist() == [3]
    assert splits["test"].tolist() == [4]
    # A source identifier repeated in the supplied official folds must not cause
    # a random reassignment of clips to hide an existing dataset limitation.
    rows[-1]["src_file"] = "1"
    assert split_indices(rows)["test"].tolist() == [4]


def test_metadata_rejects_inconsistent_target_mapping(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "audio").mkdir()
    rows = [
        {"filename": "a.wav", "fold": 1, "target": 0, "category": "dog"},
        {"filename": "b.wav", "fold": 4, "target": 0, "category": "cat"},
    ]
    with (tmp_path / "meta" / "esc50.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        (tmp_path / "audio" / row["filename"]).touch()
    with pytest.raises(ValueError, match="inconsistent"):
        read_metadata(tmp_path, expected_classes=None)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional training dependencies are absent"
)
def test_exported_onnx_logits_match_torch_and_normalization_is_embedded(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    import torch

    from hearsafe.training.cnn import ResearchCNN
    from hearsafe.training.workflow import _export

    torch.manual_seed(42)
    torch.set_num_threads(1)
    mean = np.linspace(-1, 1, 64, dtype=np.float32)
    std = np.linspace(0.5, 2, 64, dtype=np.float32)
    model = ResearchCNN(mean, std).eval()
    sample = torch.randn(1, 1, 64, 498)
    manifest = _export(model, sample, tmp_path, [f"class_{i}" for i in range(50)])
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    for scale in (0, 0.5, 2):
        inputs = sample * scale
        with torch.inference_mode():
            expected = model(inputs).numpy()
        actual = session.run(None, {"log_mel": inputs.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert session.get_inputs()[0].shape == [1, 1, 64, 498]
    assert manifest["export"]["max_logit_error"] < 1e-4
