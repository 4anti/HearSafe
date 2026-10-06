"""Watchlist filtering; model scores are not calibrated accuracy percentages."""

from .types import AlertRule, Prediction, SoundEvent


class AlertFilter:
    def __init__(self, rules: dict[str, AlertRule] | None = None):
        self.rules = dict(rules or {})
        self.reset()

    def reset(self):
        self._counts = {}
        self._low_counts = {}
        self._active = set()
        self._last = {}

    def process(self, predictions: list[Prediction], model_id: str, time_ms: float, rms: float):
        events = []
        for prediction in predictions:
            rule = self.rules.get(prediction.label_id)
            if rule is None:
                continue
            key = prediction.label_id
            strong = rms >= 0.001 and prediction.score >= rule.threshold
            if strong:
                self._counts[key] = self._counts.get(key, 0) + 1
                self._low_counts[key] = 0
            else:
                self._counts[key] = 0
                if rms < 0.001 or prediction.score < rule.threshold * 0.7:
                    self._low_counts[key] = self._low_counts.get(key, 0) + 1
                    if self._low_counts[key] >= 2:
                        self._active.discard(key)
            required = 1 if rule.transient else rule.consecutive
            cooldown = time_ms - self._last.get(key, float("-inf"))
            if strong and self._counts[key] >= required and key not in self._active:
                if cooldown >= rule.cooldown_seconds * 1000:
                    events.append(
                        SoundEvent(model_id, key, prediction.label, prediction.score, time_ms)
                    )
                    self._active.add(key)
                    self._last[key] = time_ms
        return events
