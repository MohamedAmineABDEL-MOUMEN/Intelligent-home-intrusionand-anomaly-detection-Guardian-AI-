"""IoT data processing pipeline."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import deque
from utils.config import IOT_CONFIG
from utils.logger import logger


class IoTProcessor:
    """Real-time IoT data processing pipeline."""
    
    def __init__(self, window_size=10):
        self.config = IOT_CONFIG["sensors"]
        self.window_size = window_size
        self.data_buffer = {sensor: deque(maxlen=window_size) for sensor in self.config.keys()}
        self.event_history = []
        
    def process_reading(self, reading: Dict[str, float]) -> Dict:
        """Process a single sensor reading."""
        for sensor in self.config.keys():
            if sensor in reading:
                self.data_buffer[sensor].append(reading[sensor])
        
        analysis = self._analyze_reading(reading)
        
        return analysis
    
    def _analyze_reading(self, reading: Dict[str, float]) -> Dict:
        """Analyze a reading for anomalies and events."""
        result = {
            "reading": reading,
            "alerts": [],
            "is_normal": True,
            "features": {}
        }
        
        for sensor, value in reading.items():
            if sensor == "timestamp":
                continue
                
            cfg = self.config.get(sensor, {})
            normal_min, normal_max = cfg.get("normal_range", (0, 100))
            
            if value < normal_min or value > normal_max:
                result["is_normal"] = False
                severity = self._calculate_severity(sensor, value)
                result["alerts"].append({
                    "sensor": sensor,
                    "value": value,
                    "expected_range": (normal_min, normal_max),
                    "severity": severity
                })
        
        result["features"] = self._extract_features(reading)
        
        return result
    
    def _calculate_severity(self, sensor: str, value: float) -> str:
        """Calculate severity of an anomaly."""
        cfg = self.config[sensor]
        normal_min, normal_max = cfg["normal_range"]
        sensor_range = cfg["max"] - cfg["min"]
        
        if sensor == "pir_motion":
            return "high" if value == 1 else "low"
        
        deviation = max(abs(value - normal_min), abs(value - normal_max)) / sensor_range
        
        if deviation > 0.5:
            return "critical"
        elif deviation > 0.3:
            return "high"
        elif deviation > 0.15:
            return "medium"
        return "low"
    
    def _extract_features(self, reading: Dict[str, float]) -> Dict[str, float]:
        """Extract features for ML classification."""
        features = {}
        
        for sensor in self.config.keys():
            if sensor in reading:
                value = reading[sensor]
                features[f"{sensor}_current"] = value
                
                if len(self.data_buffer[sensor]) >= 2:
                    buffer_list = list(self.data_buffer[sensor])
                    features[f"{sensor}_mean"] = np.mean(buffer_list)
                    features[f"{sensor}_std"] = np.std(buffer_list)
                    features[f"{sensor}_max"] = np.max(buffer_list)
                    features[f"{sensor}_min"] = np.min(buffer_list)
                    features[f"{sensor}_delta"] = buffer_list[-1] - buffer_list[-2]
                else:
                    features[f"{sensor}_mean"] = value
                    features[f"{sensor}_std"] = 0
                    features[f"{sensor}_max"] = value
                    features[f"{sensor}_min"] = value
                    features[f"{sensor}_delta"] = 0
        
        return features
    
    def get_feature_vector(self, reading: Dict[str, float]) -> np.ndarray:
        """Get feature vector for ML prediction."""
        features = self._extract_features(reading)
        return np.array(list(features.values())).reshape(1, -1)
    
    def get_raw_features(self, reading: Dict[str, float]) -> np.ndarray:
        """Get raw sensor values as feature vector."""
        values = [reading.get(sensor, 0) for sensor in self.config.keys()]
        return np.array(values).reshape(1, -1)
    
    def check_door_status(self, reading: Dict[str, float]) -> Tuple[bool, str]:
        """Check for unauthorized door opening."""
        vibration = reading.get("vibration", 0)
        audio = reading.get("audio", 0)
        pir = reading.get("pir_motion", 0)
        
        if vibration > 40 and audio > 65:
            if pir == 1:
                return True, "Door opened with motion detected - possible unauthorized entry"
            return True, "Door vibration detected - checking..."
        return False, "Door status: Normal"
    
    def check_suspicious_movement(self, reading: Dict[str, float]) -> Tuple[bool, str]:
        """Check for suspicious movement patterns."""
        pir = reading.get("pir_motion", 0)
        
        if len(self.data_buffer["pir_motion"]) >= 3:
            recent_motion = list(self.data_buffer["pir_motion"])[-3:]
            if sum(recent_motion) >= 2 and pir == 1:
                return True, "Sustained motion detected in monitored zone"
        
        return pir == 1, "Motion detected" if pir == 1 else "No motion"
    
    def check_device_anomaly(self, reading: Dict[str, float]) -> Tuple[bool, str]:
        """Check for abnormal device usage."""
        temp = reading.get("temperature", 20)
        co2 = reading.get("co2", 400)
        audio = reading.get("audio", 40)
        
        issues = []
        
        if temp > 35 or temp < 10:
            issues.append(f"Abnormal temperature: {temp:.1f}C")
        if co2 > 2000:
            issues.append(f"High CO2 level: {co2:.0f}ppm")
        if audio > 90:
            issues.append(f"Loud noise detected: {audio:.1f}dB")
        
        if issues:
            return True, "; ".join(issues)
        return False, "Device readings normal"
    
    def comprehensive_check(self, reading: Dict[str, float]) -> Dict:
        """Perform comprehensive security check."""
        door_alert, door_msg = self.check_door_status(reading)
        motion_alert, motion_msg = self.check_suspicious_movement(reading)
        device_alert, device_msg = self.check_device_anomaly(reading)
        
        return {
            "door": {"alert": door_alert, "message": door_msg},
            "motion": {"alert": motion_alert, "message": motion_msg},
            "device": {"alert": device_alert, "message": device_msg},
            "overall_alert": door_alert or motion_alert or device_alert
        }


if __name__ == "__main__":
    from iot.simulator import IoTSimulator
    
    processor = IoTProcessor()
    simulator = IoTSimulator()
    
    print("Testing IoT Processor...")
    for i, (reading, state) in enumerate(simulator.generate_stream(n_samples=5, include_intrusion=True)):
        print(f"\n--- Reading {i+1} (True state: {state}) ---")
        result = processor.process_reading(reading)
        check = processor.comprehensive_check(reading)
        
        print(f"Normal: {result['is_normal']}")
        print(f"Alerts: {len(result['alerts'])}")
        print(f"Door: {check['door']['message']}")
        print(f"Motion: {check['motion']['message']}")
        print(f"Device: {check['device']['message']}")
