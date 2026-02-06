"""Autoencoder-based anomaly detection (simplified version without TensorFlow)."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils.config import ANOMALY_CONFIG, SYSTEM_CONFIG
from utils.logger import logger


class AutoencoderDetector:
    """Simplified autoencoder using sklearn MLPRegressor for anomaly detection."""
    
    def __init__(self):
        self.config = ANOMALY_CONFIG["autoencoder"]
        self.encoder = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.threshold = None
        self.model_path = os.path.join(SYSTEM_CONFIG["model_dir"], "autoencoder.joblib")
        
        os.makedirs(SYSTEM_CONFIG["model_dir"], exist_ok=True)
    
    def _build_model(self, input_dim: int):
        """Build autoencoder-style model using MLP."""
        encoding_dim = self.config["encoding_dim"]
        
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(encoding_dim * 2, encoding_dim, encoding_dim * 2, input_dim),
            activation='relu',
            solver='adam',
            max_iter=self.config["epochs"],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    
    def train(self, data: np.ndarray, feature_names=None):
        """Train the autoencoder model."""
        logger.info("Training Autoencoder model...")
        
        self.feature_names = feature_names
        if feature_names is not None:
            data = pd.DataFrame(data, columns=feature_names)
            
        data_scaled = self.scaler.fit_transform(data)
        
        input_dim = data.shape[1]
        self._build_model(input_dim)
        
        self.encoder.fit(data_scaled, data_scaled)
        
        reconstructed = self.encoder.predict(data_scaled)
        reconstruction_errors = np.mean((data_scaled - reconstructed) ** 2, axis=1)
        
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        self.is_trained = True
        
        anomalies = reconstruction_errors > self.threshold
        n_anomalies = np.sum(anomalies)
        
        logger.success(f"Autoencoder trained. Threshold: {self.threshold:.4f}, Anomalies: {n_anomalies}")
        
        return {
            "n_samples": len(data),
            "threshold": float(self.threshold),
            "mean_reconstruction_error": float(np.mean(reconstruction_errors)),
            "max_reconstruction_error": float(np.max(reconstruction_errors)),
            "n_anomalies": int(n_anomalies)
        }
    
    def predict(self, data: np.ndarray) -> tuple:
        """Predict if data is anomalous based on reconstruction error."""
        if not self.is_trained:
            return False, 0.5
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            data = pd.DataFrame(data, columns=self.feature_names)
            
        data_scaled = self.scaler.transform(data)
        
        reconstructed = self.encoder.predict(data_scaled)
        reconstruction_error = np.mean((data_scaled - reconstructed) ** 2, axis=1)[0]
        
        is_anomaly = reconstruction_error > self.threshold
        
        anomaly_score = min(reconstruction_error / (self.threshold * 2), 1.0)
        
        return bool(is_anomaly), float(anomaly_score)
    
    def get_reconstruction_error(self, data: np.ndarray) -> float:
        """Get reconstruction error for data."""
        if not self.is_trained:
            return 0.0
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        data_scaled = self.scaler.transform(data)
        reconstructed = self.encoder.predict(data_scaled)
        
        return float(np.mean((data_scaled - reconstructed) ** 2))
    
    def save(self):
        """Save model to disk."""
        if not self.is_trained:
            return
        
        model_data = {
            "encoder": self.encoder,
            "scaler": self.scaler,
            "threshold": self.threshold
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Autoencoder saved to {self.model_path}")
    
    def load(self) -> bool:
        """Load model from disk."""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.encoder = model_data["encoder"]
            self.scaler = model_data["scaler"]
            self.threshold = model_data["threshold"]
            self.is_trained = True
            logger.info("Autoencoder loaded successfully.")
            return True
        return False


if __name__ == "__main__":
    from iot.simulator import IoTSimulator
    
    simulator = IoTSimulator()
    data = simulator.generate_dataset(n_samples=500, include_intrusion=False)
    X = data.drop(columns=["label"]).values
    
    detector = AutoencoderDetector()
    results = detector.train(X)
    
    print("\nTraining Results:")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Threshold: {results['threshold']:.4f}")
    print(f"  Mean Error: {results['mean_reconstruction_error']:.4f}")
    
    test_data = simulator.generate_dataset(n_samples=10, include_intrusion=True)
    X_test = test_data.drop(columns=["label"]).values
    
    print("\nTest predictions:")
    for i, row in enumerate(X_test):
        is_anomaly, score = detector.predict(row)
        print(f"  Sample {i}: Anomaly={is_anomaly}, Score={score:.3f}")
    
    detector.save()
