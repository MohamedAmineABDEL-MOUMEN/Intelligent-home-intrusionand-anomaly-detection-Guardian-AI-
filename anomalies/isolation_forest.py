"""Isolation Forest anomaly detection."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils.config import ANOMALY_CONFIG, SYSTEM_CONFIG
from utils.logger import logger


class IsolationForestDetector:
    """Isolation Forest based anomaly detector."""
    
    def __init__(self):
        self.config = ANOMALY_CONFIG["isolation_forest"]
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = os.path.join(SYSTEM_CONFIG["model_dir"], "isolation_forest.joblib")
        
        os.makedirs(SYSTEM_CONFIG["model_dir"], exist_ok=True)
    
    def train(self, data: np.ndarray, feature_names=None):
        """Train the Isolation Forest model."""
        logger.info("Training Isolation Forest model...")
        
        self.feature_names = feature_names
        if feature_names is not None:
            data = pd.DataFrame(data, columns=feature_names)
            
        data_scaled = self.scaler.fit_transform(data)
        
        self.model = IsolationForest(
            n_estimators=self.config["n_estimators"],
            contamination=self.config["contamination"],
            random_state=self.config["random_state"],
            n_jobs=-1
        )
        
        self.model.fit(data_scaled)
        self.is_trained = True
        
        scores = self.model.decision_function(data_scaled)
        predictions = self.model.predict(data_scaled)
        
        n_anomalies = np.sum(predictions == -1)
        logger.success(f"Isolation Forest trained. Detected {n_anomalies} anomalies in training data.")
        
        return {
            "n_samples": len(data),
            "n_anomalies": int(n_anomalies),
            "anomaly_rate": float(n_anomalies / len(data)),
            "score_range": (float(scores.min()), float(scores.max()))
        }
    
    def predict(self, data: np.ndarray) -> tuple:
        """Predict if data is anomalous."""
        if not self.is_trained:
            return False, 0.5
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            data = pd.DataFrame(data, columns=self.feature_names)
            
        data_scaled = self.scaler.transform(data)
        
        prediction = self.model.predict(data_scaled)[0]
        score = self.model.decision_function(data_scaled)[0]
        
        anomaly_score = 1 - (score + 0.5)
        anomaly_score = np.clip(anomaly_score, 0, 1)
        
        is_anomaly = prediction == -1
        
        return is_anomaly, float(anomaly_score)
    
    def save(self):
        """Save model to disk."""
        if not self.is_trained:
            return
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Isolation Forest saved to {self.model_path}")
    
    def load(self) -> bool:
        """Load model from disk."""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.is_trained = True
            logger.info("Isolation Forest loaded successfully.")
            return True
        return False


if __name__ == "__main__":
    from iot.simulator import IoTSimulator
    
    simulator = IoTSimulator()
    data = simulator.generate_dataset(n_samples=500)
    X = data.drop(columns=["label"]).values
    
    detector = IsolationForestDetector()
    results = detector.train(X)
    
    print("\nTraining Results:")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Anomalies detected: {results['n_anomalies']}")
    print(f"  Anomaly rate: {results['anomaly_rate']:.2%}")
    
    test_normal = X[0].reshape(1, -1)
    is_anomaly, score = detector.predict(test_normal)
    print(f"\nTest prediction: Anomaly={is_anomaly}, Score={score:.3f}")
    
    detector.save()
