"""Configuration settings for the intrusion detection system."""

IOT_CONFIG = {
    "sensors": {
        "vibration": {"min": 0, "max": 100, "normal_range": (0, 30), "unit": "m/s2"},
        "audio": {"min": 0, "max": 120, "normal_range": (20, 60), "unit": "dB"},
        "temperature": {"min": -10, "max": 50, "normal_range": (18, 28), "unit": "C"},
        "co2": {"min": 300, "max": 5000, "normal_range": (400, 1000), "unit": "ppm"},
        "pir_motion": {"min": 0, "max": 1, "normal_range": (0, 0), "unit": "binary"}
    },
    "sampling_rate": 1.0,
    "anomaly_threshold": 0.7
}

CAMERA_CONFIG = {
    "resolution": (640, 480),
    "fps": 15,
    "recording_duration": 10,
    "confidence_threshold": 0.5,
    "tracking_enabled": True
}

ML_CONFIG = {
    "models": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "svm": {
            "kernel": "rbf",
            "C": 0.5,
            "gamma": 0.1
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    "test_size": 0.2,
    "random_state": 42
}

ANOMALY_CONFIG = {
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": 0.1,
        "random_state": 42
    },
    "autoencoder": {
        "encoding_dim": 8,
        "epochs": 50,
        "batch_size": 32
    },
    "lstm": {
        "sequence_length": 10,
        "hidden_units": 32,
        "epochs": 50,
        "batch_size": 32
    }
}

SYSTEM_CONFIG = {
    "log_dir": "logs",
    "output_dir": "outputs",
    "model_dir": "models/saved",
    "data_dir": "data"
}
