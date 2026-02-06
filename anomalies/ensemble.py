"""Ensemble anomaly detection combining multiple methods."""
import numpy as np
from typing import Dict, Tuple
from anomalies.isolation_forest import IsolationForestDetector
from anomalies.autoencoder import AutoencoderDetector
from anomalies.lstm_detector import LSTMDetector
from utils.logger import logger


class AnomalyEnsemble:
    """Ensemble of anomaly detection models."""
    
    def __init__(self):
        self.isolation_forest = IsolationForestDetector()
        self.autoencoder = AutoencoderDetector()
        self.lstm = LSTMDetector()
        self.is_trained = False
        
    def train_all(self, point_data: np.ndarray, sequence_data: np.ndarray = None):
        """Train all anomaly detection models."""
        logger.info("Training Anomaly Detection Ensemble...")
        
        results = {}
        
        logger.info("Training Isolation Forest...")
        results["isolation_forest"] = self.isolation_forest.train(point_data)
        
        logger.info("Training Autoencoder...")
        results["autoencoder"] = self.autoencoder.train(point_data)
        
        if sequence_data is not None:
            logger.info("Training LSTM detector...")
            results["lstm"] = self.lstm.train(sequence_data)
        
        self.is_trained = True
        logger.success("Anomaly Detection Ensemble trained successfully!")
        
        return results
    
    def predict_point(self, data: np.ndarray) -> Dict:
        """Predict anomaly using point-based detectors."""
        results = {
            "isolation_forest": {"is_anomaly": False, "score": 0.5},
            "autoencoder": {"is_anomaly": False, "score": 0.5},
            "ensemble": {"is_anomaly": False, "score": 0.5, "votes": 0}
        }
        
        if self.isolation_forest.is_trained:
            is_anomaly, score = self.isolation_forest.predict(data)
            results["isolation_forest"] = {"is_anomaly": is_anomaly, "score": score}
        
        if self.autoencoder.is_trained:
            is_anomaly, score = self.autoencoder.predict(data)
            results["autoencoder"] = {"is_anomaly": is_anomaly, "score": score}
        
        votes = sum([
            results["isolation_forest"]["is_anomaly"],
            results["autoencoder"]["is_anomaly"]
        ])
        
        avg_score = np.mean([
            results["isolation_forest"]["score"],
            results["autoencoder"]["score"]
        ])
        
        results["ensemble"] = {
            "is_anomaly": votes >= 1,
            "score": avg_score,
            "votes": votes
        }
        
        return results
    
    def predict_sequence(self, sequence: np.ndarray) -> Dict:
        """Predict anomaly in a sequence."""
        results = {"lstm": {"is_anomaly": False, "score": 0.5, "errors": []}}
        
        if self.lstm.is_trained:
            is_anomaly, score, errors = self.lstm.predict_sequence(sequence)
            results["lstm"] = {
                "is_anomaly": is_anomaly,
                "score": score,
                "errors": errors
            }
        
        return results
    
    def predict_combined(self, point_data: np.ndarray, sequence_data: np.ndarray = None) -> Dict:
        """Combined prediction using all available detectors."""
        point_results = self.predict_point(point_data)
        
        if sequence_data is not None and self.lstm.is_trained:
            seq_results = self.predict_sequence(sequence_data)
            point_results["lstm"] = seq_results["lstm"]
            
            all_anomalies = [
                point_results["isolation_forest"]["is_anomaly"],
                point_results["autoencoder"]["is_anomaly"],
                point_results["lstm"]["is_anomaly"]
            ]
            all_scores = [
                point_results["isolation_forest"]["score"],
                point_results["autoencoder"]["score"],
                point_results["lstm"]["score"]
            ]
            
            point_results["ensemble"] = {
                "is_anomaly": sum(all_anomalies) >= 2,
                "score": np.mean(all_scores),
                "votes": sum(all_anomalies)
            }
        
        return point_results
    
    def save_all(self):
        """Save all models."""
        self.isolation_forest.save()
        self.autoencoder.save()
        self.lstm.save()
        logger.success("All anomaly detection models saved.")
    
    def load_all(self) -> bool:
        """Load all models."""
        success = True
        success &= self.isolation_forest.load()
        success &= self.autoencoder.load()
        success &= self.lstm.load()
        
        if success:
            self.is_trained = True
            logger.success("All anomaly detection models loaded.")
        
        return success


if __name__ == "__main__":
    from iot.simulator import IoTSimulator
    
    simulator = IoTSimulator()
    
    point_data = simulator.generate_dataset(n_samples=500, include_intrusion=False)
    X_point = point_data.drop(columns=["label"]).values
    
    sequences, _ = simulator.generate_sequence_data(n_sequences=100, sequence_length=10, include_anomalies=False)
    
    ensemble = AnomalyEnsemble()
    results = ensemble.train_all(X_point, sequences)
    
    print("\n" + "="*60)
    print("ANOMALY ENSEMBLE TRAINING RESULTS")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        for key, value in model_results.items():
            print(f"  {key}: {value}")
    
    test_data = simulator.generate_dataset(n_samples=10, include_intrusion=True)
    X_test = test_data.drop(columns=["label"]).values
    
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTIONS")
    print("="*60)
    
    for i, row in enumerate(X_test[:5]):
        pred = ensemble.predict_point(row)
        print(f"\nSample {i}:")
        print(f"  Isolation Forest: Anomaly={pred['isolation_forest']['is_anomaly']}, Score={pred['isolation_forest']['score']:.3f}")
        print(f"  Autoencoder: Anomaly={pred['autoencoder']['is_anomaly']}, Score={pred['autoencoder']['score']:.3f}")
        print(f"  Ensemble: Anomaly={pred['ensemble']['is_anomaly']}, Score={pred['ensemble']['score']:.3f}, Votes={pred['ensemble']['votes']}")
    
    ensemble.save_all()
