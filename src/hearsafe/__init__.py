"""HearSafe's reusable, local sound detector."""

from .detector import Detector
from .models import load_model
from .types import AlertRule, AnalysisFrame, Prediction, SoundEvent

__version__ = "0.1.0"
__all__ = ["Detector", "load_model", "AlertRule", "AnalysisFrame", "Prediction", "SoundEvent"]
