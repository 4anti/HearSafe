from hearsafe.alerts import AlertFilter
from hearsafe.types import AlertRule, Prediction


def test_sustained_once_and_rearming():
    policy = AlertFilter({"a": AlertRule(cooldown_seconds=1)})
    prediction = [Prediction("a", "Alarm", 0.9)]
    assert policy.process(prediction, "model", 0, 0.1) == []
    assert len(policy.process(prediction, "model", 500, 0.1)) == 1
    assert policy.process(prediction, "model", 10000, 0.1) == []
    low = [Prediction("a", "Alarm", 0.1)]
    policy.process(low, "model", 11000, 0.1)
    policy.process(low, "model", 11500, 0.1)
    assert policy.process(prediction, "model", 12000, 0.1) == []
    assert len(policy.process(prediction, "model", 12500, 0.1)) == 1


def test_transient_overlapping_and_silence():
    policy = AlertFilter({"a": AlertRule(transient=True), "b": AlertRule(transient=True)})
    prediction = [Prediction("a", "Knock", 0.9), Prediction("b", "Shatter", 0.8)]
    assert policy.process(prediction, "model", 0, 0) == []
    assert len(policy.process(prediction, "model", 500, 0.1)) == 2
    policy.reset()
    assert len(policy.process(prediction, "model", 1000, 0.1)) == 2


def test_disabled_and_rule_validation():
    import pytest

    assert AlertFilter().process([Prediction("a", "Alarm", 0.99)], "m", 0, 1) == []
    with pytest.raises(ValueError):
        AlertRule(threshold=0)
