"""
Smart Anti-Intrusion System for Connected Homes
Main entry point and system orchestration.
"""
import os
import sys
import time
import argparse
from colorama import Fore, Style, init

init(autoreset=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger, NotificationLogger
from utils.config import SYSTEM_CONFIG
from iot.simulator import IoTSimulator
from iot.processor import IoTProcessor
from models.classifier import IntrusionClassifier
from camera.detector import CameraDetector
from anomalies.ensemble import AnomalyEnsemble


def ensure_directories():
    """Create necessary directories."""
    dirs = [
        SYSTEM_CONFIG["log_dir"],
        SYSTEM_CONFIG["output_dir"],
        SYSTEM_CONFIG["model_dir"],
        SYSTEM_CONFIG["data_dir"],
        "outputs/recordings"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_banner():
    """Print system banner."""
    banner = f"""
{Fore.CYAN}{'='*70}
     SMART ANTI-INTRUSION SYSTEM FOR CONNECTED HOMES
              IoT + Video Recognition + ML
{'='*70}{Style.RESET_ALL}

{Fore.YELLOW}Features:{Style.RESET_ALL}
  - Real-time IoT sensor monitoring (vibration, audio, temp, CO2, PIR)
  - ML-based event classification (Random Forest, SVM, XGBoost)
  - Camera-based human detection (YOLOv8-style) with DeepSort tracking
  - Anomaly detection (Isolation Forest, Autoencoder, LSTM)
  - Automatic video recording on intrusion detection

{Fore.GREEN}Status: System Initializing...{Style.RESET_ALL}
"""
    print(banner)


def train_models():
    """Train all ML models."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("TRAINING MACHINE LEARNING MODELS")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    simulator = IoTSimulator()
    
    print("1. Generating training data...")
    point_data = simulator.generate_dataset(n_samples=2000, include_intrusion=True)
    sequences, _ = simulator.generate_sequence_data(n_sequences=200, sequence_length=10, include_anomalies=False)
    
    print(f"   - Point data: {point_data.shape[0]} samples")
    print(f"   - Sequence data: {sequences.shape[0]} sequences")
    
    print("\n2. Training Classification Models...")
    classifier = IntrusionClassifier()
    class_results = classifier.train(point_data)
    classifier.save_models()
    
    print("\n   Results:")
    for model_name, result in class_results.items():
        print(f"   - {model_name}: Accuracy = {result['accuracy']:.4f}")
    
    print("\n3. Training Anomaly Detection Models...")
    X_point = point_data.drop(columns=["label"]).values
    ensemble = AnomalyEnsemble()
    anom_results = ensemble.train_all(X_point, sequences)
    ensemble.save_all()
    
    print("\n   Results:")
    for model_name, result in anom_results.items():
        if "threshold" in result:
            print(f"   - {model_name}: Threshold = {result['threshold']:.4f}")
    
    print(f"\n{Fore.GREEN}All models trained and saved successfully!{Style.RESET_ALL}")
    
    return classifier, ensemble


def run_iot_demo(classifier, anomaly_ensemble, duration=30):
    """Run IoT sensor monitoring demo."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("IOT SENSOR MONITORING DEMO")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    simulator = IoTSimulator()
    processor = IoTProcessor()
    
    intrusion_count = 0
    anomaly_count = 0
    
    print(f"Monitoring for {duration} sensor readings...\n")
    
    for i, (reading, true_state) in enumerate(simulator.generate_stream(n_samples=duration, include_intrusion=True)):
        result = processor.process_reading(reading)
        check = processor.comprehensive_check(reading)
        
        features = processor.get_raw_features(reading)
        prediction, confidence = classifier.predict(features, model_name="xgboost")
        
        anomaly_result = anomaly_ensemble.predict_point(features)
        
        status_color = Fore.GREEN if prediction == 0 else Fore.RED
        predicted_label = ["Normal", "Intrusion", "Anomaly"][prediction]
        
        print(f"[{i+1:3d}] {status_color}Status: {predicted_label} (conf: {confidence:.2f}){Style.RESET_ALL} | "
              f"True: {true_state} | Anomaly: {anomaly_result['ensemble']['is_anomaly']}")
        
        if check["overall_alert"]:
            if check["door"]["alert"]:
                logger.warning(f"   Door: {check['door']['message']}")
            if check["motion"]["alert"]:
                logger.warning(f"   Motion: {check['motion']['message']}")
            if check["device"]["alert"]:
                logger.warning(f"   Device: {check['device']['message']}")
        
        if prediction == 1:
            intrusion_count += 1
            logger.mobile_notification(
                "INTRUSION ALERT",
                f"Suspicious activity detected! Confidence: {confidence:.0%}",
                priority="high"
            )
        
        if anomaly_result['ensemble']['is_anomaly']:
            anomaly_count += 1
        
        time.sleep(0.1)
    
    print(f"\n{Fore.YELLOW}Demo Summary:{Style.RESET_ALL}")
    print(f"  Total readings: {duration}")
    print(f"  Intrusions detected: {intrusion_count}")
    print(f"  Anomalies detected: {anomaly_count}")


def run_camera_demo():
    """Run camera-based detection demo."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("CAMERA-BASED INTRUSION DETECTION DEMO")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    detector = CameraDetector()
    
    print("Running simulation with 100 frames...")
    print("Intrusion occurs between frames 30-60\n")
    
    results = detector.run_simulation(n_frames=100, intrusion_start=30, intrusion_end=60)
    
    detector.save_sample_frames(results)
    
    intrusion_frames = sum(1 for r in results if r["is_intrusion"])
    
    print(f"\n{Fore.YELLOW}Camera Demo Summary:{Style.RESET_ALL}")
    print(f"  Total frames: {len(results)}")
    print(f"  Intrusion frames: {intrusion_frames}")
    print(f"  Recordings saved: {detector.intrusion_count}")
    print(f"  Sample frames saved to: outputs/")


def run_full_system():
    """Run the complete integrated system."""
    print_banner()
    ensure_directories()
    
    print(f"\n{Fore.YELLOW}Phase 1: Training Models{Style.RESET_ALL}")
    print("-" * 40)
    classifier, anomaly_ensemble = train_models()
    
    print(f"\n{Fore.YELLOW}Phase 2: IoT Monitoring Demo{Style.RESET_ALL}")
    print("-" * 40)
    run_iot_demo(classifier, anomaly_ensemble, duration=20)
    
    print(f"\n{Fore.YELLOW}Phase 3: Camera Detection Demo{Style.RESET_ALL}")
    print("-" * 40)
    run_camera_demo()
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print("SYSTEM DEMONSTRATION COMPLETE")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    print(f"""
{Fore.CYAN}Outputs generated:{Style.RESET_ALL}
  - Trained models saved in: models/saved/
  - Detection frames saved in: outputs/
  - Video recordings saved in: outputs/recordings/
  - System logs saved in: logs/

{Fore.CYAN}Available commands:{Style.RESET_ALL}
  python main.py --train     : Train all models
  python main.py --iot       : Run IoT monitoring demo
  python main.py --camera    : Run camera detection demo
  python main.py --full      : Run complete system demo (default)

{Fore.GREEN}System Status: HOME SECURED{Style.RESET_ALL}
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smart Anti-Intrusion System")
    parser.add_argument("--train", action="store_true", help="Train all ML models")
    parser.add_argument("--iot", action="store_true", help="Run IoT monitoring demo")
    parser.add_argument("--camera", action="store_true", help="Run camera detection demo")
    parser.add_argument("--full", action="store_true", help="Run complete system demo")
    
    args = parser.parse_args()
    
    ensure_directories()
    
    if args.train:
        print_banner()
        train_models()
    elif args.iot:
        print_banner()
        classifier = IntrusionClassifier()
        if not classifier.load_models():
            print("Models not found. Training first...")
            classifier, anomaly_ensemble = train_models()
        else:
            anomaly_ensemble = AnomalyEnsemble()
            anomaly_ensemble.load_all()
        run_iot_demo(classifier, anomaly_ensemble)
    elif args.camera:
        print_banner()
        run_camera_demo()
    else:
        run_full_system()


if __name__ == "__main__":
    main()
