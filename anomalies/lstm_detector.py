"""LSTM-based temporal anomaly detection (simplified using scikit-learn)."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils.config import ANOMALY_CONFIG, SYSTEM_CONFIG
from utils.logger import logger


class LSTMDetector:
    """Simplified temporal anomaly detector using MLP (simulating LSTM behavior)."""
    
    def __init__(self):
        self.config = ANOMALY_CONFIG["lstm"]
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.threshold = None
        self.sequence_length = self.config["sequence_length"]
        self.model_path = os.path.join(SYSTEM_CONFIG["model_dir"], "lstm_detector.joblib")
        
        os.makedirs(SYSTEM_CONFIG["model_dir"], exist_ok=True)
    
    def _prepare_sequences(self, data: np.ndarray) -> tuple:
        """Prepare sequences for training (input: sequence, output: next step)."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length].flatten()
            target = data[i + self.sequence_length]
            X.append(sequence)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self, sequences: np.ndarray, feature_names=None):
        """Train the temporal anomaly detector."""
        logger.info("Training LSTM-style temporal detector...")
        
        self.feature_names = feature_names
        if sequences.ndim == 2:
            sequences = sequences.reshape(-1, self.sequence_length, sequences.shape[1] // self.sequence_length)
        
        flattened = sequences.reshape(len(sequences), -1)
        
        if feature_names is not None:
            # Create multi-step feature names if provided
            flat_feature_names = [f"{name}_t-{t}" for t in range(self.sequence_length) for name in feature_names]
            flattened_df = pd.DataFrame(flattened, columns=flat_feature_names)
            flattened_scaled = self.scaler.fit_transform(flattened_df)
        else:
            flattened_scaled = self.scaler.fit_transform(flattened)
            
        sequences_scaled = flattened_scaled.reshape(sequences.shape)
        
        X, y = [], []
        for seq in sequences_scaled:
            for i in range(len(seq) - 1):
                input_window = seq[:i+1].flatten()
                input_window = np.pad(input_window, (0, max(0, self.sequence_length * seq.shape[1] - len(input_window))), 
                                     mode='constant')[:self.sequence_length * seq.shape[1]]
                X.append(input_window)
                y.append(seq[i+1])
        
        X = np.array(X)
        y = np.array(y)
        
        hidden_units = self.config["hidden_units"]
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_units * 2, hidden_units, hidden_units // 2),
            activation='tanh',
            solver='adam',
            max_iter=self.config["epochs"],
            random_state=42,
            early_stopping=True
        )
        
        self.model.fit(X, y)
        
        predictions = self.model.predict(X)
        errors = np.mean((y - predictions) ** 2, axis=1)
        self.threshold = np.percentile(errors, 95)
        
        self.is_trained = True
        
        logger.success(f"Temporal detector trained. Threshold: {self.threshold:.4f}")
        
        return {
            "n_sequences": len(sequences),
            "threshold": float(self.threshold),
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors))
        }
    
    def predict_sequence(self, sequence: np.ndarray) -> tuple:
        """Predict if a sequence contains temporal anomalies."""
        if not self.is_trained:
            return False, 0.5, []
        
        if sequence.ndim == 1:
            n_features = len(sequence) // self.sequence_length
            sequence = sequence.reshape(self.sequence_length, n_features)
        
        flattened = sequence.flatten().reshape(1, -1)
        
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            flat_feature_names = [f"{name}_t-{t}" for t in range(self.sequence_length) for name in self.feature_names]
            flattened_df = pd.DataFrame(flattened, columns=flat_feature_names)
            flattened_scaled = self.scaler.transform(flattened_df)
        else:
            flattened_scaled = self.scaler.transform(flattened)
            
        sequence_scaled = flattened_scaled.reshape(sequence.shape)
        
        errors = []
        for i in range(len(sequence_scaled) - 1):
            input_window = sequence_scaled[:i+1].flatten()
            input_window = np.pad(input_window, (0, max(0, self.sequence_length * sequence.shape[1] - len(input_window))),
                                 mode='constant')[:self.sequence_length * sequence.shape[1]]
            input_window = input_window.reshape(1, -1)
            
            prediction = self.model.predict(input_window)
            error = np.mean((sequence_scaled[i+1] - prediction) ** 2)
            errors.append(error)
        
        if not errors:
            return False, 0.5, []
        
        max_error = max(errors)
        is_anomaly = max_error > self.threshold
        anomaly_score = min(max_error / (self.threshold * 2), 1.0)
        
        return bool(is_anomaly), float(anomaly_score), errors
    
    def save(self):
        """Save model to disk."""
        if not self.is_trained:
            return
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "sequence_length": self.sequence_length
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Temporal detector saved to {self.model_path}")
    
    def load(self) -> bool:
        """Load model from disk."""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.threshold = model_data["threshold"]
            self.sequence_length = model_data["sequence_length"]
            self.is_trained = True
            logger.info("Temporal detector loaded successfully.")
            return True
        return False


if __name__ == "__main__":
    from iot.simulator import IoTSimulator
    
    simulator = IoTSimulator()
    sequences, labels = simulator.generate_sequence_data(n_sequences=200, sequence_length=10)
    
    normal_sequences = sequences[labels == 0]
    
    detector = LSTMDetector()
    results = detector.train(normal_sequences)
    
    print("\nTraining Results:")
    print(f"  Sequences: {results['n_sequences']}")
    print(f"  Threshold: {results['threshold']:.4f}")
    print(f"  Mean Error: {results['mean_error']:.4f}")
    
    test_sequences, test_labels = simulator.generate_sequence_data(n_sequences=5, sequence_length=10)
    
    print("\nTest predictions:")
    for i, (seq, label) in enumerate(zip(test_sequences, test_labels)):
        is_anomaly, score, _ = detector.predict_sequence(seq)
        print(f"  Seq {i}: Actual={label}, Predicted Anomaly={is_anomaly}, Score={score:.3f}")
    
    detector.save()
