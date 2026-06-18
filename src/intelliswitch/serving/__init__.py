from .predict import (
    predict_recording, InferenceResult, LocationPrediction, WheelSetPrediction,
)

# backward-compatible alias
SensorPrediction = LocationPrediction

__all__ = [
    "predict_recording", "InferenceResult",
    "LocationPrediction", "WheelSetPrediction", "SensorPrediction",
]
