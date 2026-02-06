"""Simulated IoT sensor data generator."""
import numpy as np
import pandas as pd
import time
from typing import Generator, Dict, Tuple
from utils.config import IOT_CONFIG


class IoTSimulator:
    """Simulates IoT sensor data for testing the intrusion detection system."""
    
    def __init__(self, seed=None):
        # Use None seed for real randomness during live testing
        self.rng = np.random.default_rng(seed)
        self.config = IOT_CONFIG["sensors"]
        self.current_state = "normal"
        self.intrusion_probability = 0.1
        
    def _generate_normal_reading(self, sensor: str) -> float:
        """Generate a normal reading for a sensor with added noise."""
        cfg = self.config[sensor]
        min_val, max_val = cfg["normal_range"]
        
        if sensor == "pir_motion":
            # 5% chance of false positive in normal state
            return float(self.rng.choice([0, 1], p=[0.95, 0.05]))
        else:
            mean = (min_val + max_val) / 2
            # Add some jitter to the normal range
            std = (max_val - min_val) / 4
            value = self.rng.normal(mean, std)
            # Add extra high-frequency noise
            noise = self.rng.uniform(-0.5, 0.5)
            return float(np.clip(value + noise, cfg["min"], cfg["max"]))
    
    def _generate_intrusion_reading(self, sensor: str) -> float:
        """Generate an intrusion-indicating reading with high variability."""
        cfg = self.config[sensor]
        normal_min, normal_max = cfg["normal_range"]
        
        # Increase variability by using different distributions and ranges
        if sensor == "vibration":
            # Dynamic vibration: sometimes subtle, sometimes violent
            return float(self.rng.uniform(35, 110))
        elif sensor == "audio":
            # Burst of sound
            return float(self.rng.uniform(65, 130))
        elif sensor == "temperature":
            # Sudden heat or cold (fire or window open)
            return float(self.rng.choice([
                self.rng.uniform(cfg["min"], normal_min - 1),
                self.rng.uniform(normal_max + 2, cfg["max"])
            ]))
        elif sensor == "co2":
            # Gradual or sudden CO2 spike
            return float(self.rng.uniform(1100, 4000))
        elif sensor == "pir_motion":
            # Intrusion usually means motion, but maybe not 100% of the time (stealth)
            return float(self.rng.choice([1.0, 0.0], p=[0.9, 0.1]))
        return self._generate_normal_reading(sensor)
    
    def _generate_anomaly_reading(self, sensor: str) -> float:
        """Generate an anomalous reading (random spikes/drops)."""
        cfg = self.config[sensor]
        
        if sensor == "pir_motion":
            return float(self.rng.choice([0, 1], p=[0.4, 0.6]))
        else:
            anomaly_type = self.rng.choice(["spike", "drop", "drift", "noise"])
            normal_min, normal_max = cfg["normal_range"]
            
            if anomaly_type == "spike":
                return float(self.rng.uniform(normal_max * 1.1, cfg["max"]))
            elif anomaly_type == "drop":
                return float(self.rng.uniform(cfg["min"], normal_min * 0.9))
            elif anomaly_type == "drift":
                # Simulate a drifting sensor
                return float(self.rng.normal(normal_max * 1.05, 2.0))
            else:
                # Heavy noise
                return float(self.rng.uniform(cfg["min"], cfg["max"]))
    
    def generate_single_reading(self, include_intrusion=False) -> Tuple[Dict[str, float], str]:
        """Generate a single reading from all sensors."""
        if include_intrusion:
            roll = self.rng.random()
            if roll < 0.4: # Reduced normal chance
                self.current_state = "normal"
            elif roll < 0.7:
                self.current_state = "anomaly"
            else:
                self.current_state = "intrusion"
        else:
            self.current_state = "normal"
        
        reading = {"timestamp": time.time()}
        
        for sensor in self.config.keys():
            if self.current_state == "normal":
                reading[sensor] = self._generate_normal_reading(sensor)
            elif self.current_state == "intrusion":
                reading[sensor] = self._generate_intrusion_reading(sensor)
            else:
                reading[sensor] = self._generate_anomaly_reading(sensor)
        
        return reading, self.current_state
    
    def generate_stream(self, n_samples=100, include_intrusion=True) -> Generator:
        """Generate a stream of sensor readings."""
        for _ in range(n_samples):
            reading, state = self.generate_single_reading(include_intrusion)
            yield reading, state
            time.sleep(0.01)
    
    def generate_dataset(self, n_samples=1000, include_intrusion=True) -> pd.DataFrame:
        """Generate a labeled dataset for training with high entropy."""
        data = []
        labels = []
        
        for _ in range(n_samples):
            reading, state = self.generate_single_reading(include_intrusion)
            reading_values = {k: v for k, v in reading.items() if k != "timestamp"}
            data.append(reading_values)
            labels.append(0 if state == "normal" else (1 if state == "intrusion" else 2))
        
        df = pd.DataFrame(data)
        df["label"] = labels
        # Shuffle to avoid sequence bias
        df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def generate_sequence_data(self, n_sequences=100, sequence_length=10, include_anomalies=True) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sequence data for LSTM/autoencoder training."""
        all_sequences = []
        all_labels = []
        
        for _ in range(n_sequences):
            is_anomaly = include_anomalies and self.rng.random() < 0.2
            sequence = []
            
            for _ in range(sequence_length):
                if is_anomaly:
                    reading, _ = self.generate_single_reading(include_intrusion=True)
                else:
                    reading, _ = self.generate_single_reading(include_intrusion=False)
                sequence.append([reading[s] for s in self.config.keys()])
            
            all_sequences.append(sequence)
            all_labels.append(1 if is_anomaly else 0)
        
        return np.array(all_sequences), np.array(all_labels)
