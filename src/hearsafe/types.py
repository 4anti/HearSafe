"""Public results and model metadata; timestamps are relative to the audio session."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Label:
    id: str
    name: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    name: str
    backend: str
    sample_rate: int
    window_samples: int
    hop_samples: int
    labels: list[Label]
    model_file: str
    sha256: str
    frontend: str = "waveform"
    license: str = ""
    provenance: str | dict = ""
    schema_version: int = 1


@dataclass(frozen=True)
class Prediction:
    label_id: str
    label: str
    score: float


@dataclass(frozen=True)
class SoundEvent:
    model_id: str
    label_id: str
    label: str
    score: float
    time_ms: float
    state: str = "started"
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisFrame:
    model_id: str
    start_ms: float
    end_ms: float
    predictions: list[Prediction]
    rms: float
    inference_ms: float
    events: list[SoundEvent] = field(default_factory=list)
    schema_version: int = 1

    def top(self, k: int = 5) -> list[Prediction]:
        return sorted(self.predictions, key=lambda item: item.score, reverse=True)[:k]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlertRule:
    threshold: float = 0.5
    transient: bool = False
    cooldown_seconds: float = 5.0
    consecutive: int = 2

    def __post_init__(self):
        if not 0 < self.threshold <= 1:
            raise ValueError("Alert threshold must be greater than 0 and at most 1")
        if self.cooldown_seconds < 0 or self.consecutive < 1:
            raise ValueError("Cooldown must be nonnegative and consecutive count positive")
