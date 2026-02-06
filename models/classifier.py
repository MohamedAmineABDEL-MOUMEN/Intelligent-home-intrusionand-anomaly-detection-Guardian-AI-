"""ML classification models for event detection."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
from utils.config import ML_CONFIG, SYSTEM_CONFIG
from utils.logger import logger


class IntrusionClassifier:
    """Multi-model classifier for intrusion detection."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.config = ML_CONFIG
        self.model_dir = SYSTEM_CONFIG["model_dir"]
        self.is_trained = False
        self.feature_names = None
        
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _create_models(self):
        """Create fresh model instances."""
        rf_cfg = self.config["models"]["random_forest"]
        self.models["random_forest"] = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"]
        )
        
        svm_cfg = self.config["models"]["svm"]
        self.models["svm"] = SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            probability=True
        )
        
        xgb_cfg = self.config["models"]["xgboost"]
        self.models["xgboost"] = xgb.XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            random_state=xgb_cfg["random_state"],
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
    
    def train(self, data: pd.DataFrame, label_column="label"):
        """Train all models on the provided data."""
        logger.info("Starting model training...")
        
        self._create_models()
        
        X = data.drop(columns=[label_column])
        y = data[label_column]
        self.feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
            stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            if name == "svm":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                "accuracy": accuracy,
                "report": report,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.success(f"{name} trained - Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return results
    
    def predict(self, features, model_name="xgboost") -> tuple:
        """Make prediction using specified model."""
        if not self.is_trained:
            logger.warning("Models not trained. Returning default prediction.")
            return 0, 0.5
        
        model = self.models.get(model_name)
        if model is None:
            logger.warning(f"Model {model_name} not found. Using xgboost.")
            model = self.models["xgboost"]
            model_name = "xgboost"
        
        # Ensure features is a DataFrame for consistent feature names
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = pd.DataFrame(features, columns=self.feature_names)
        elif isinstance(features, list):
            features = pd.DataFrame([features], columns=self.feature_names)

        if model_name == "svm":
            features_scaled = self.scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
        
        confidence = float(max(probabilities))
        
        return int(prediction), confidence
    
    def predict_ensemble(self, features) -> dict:
        """Make prediction using all models and return comparison."""
        if not self.is_trained:
            return {"final_prediction": 0, "avg_confidence": 0.5, "models": {}}
        
        results = {}
        predictions = []
        confidences = []
        
        for name in self.models.keys():
            pred, conf = self.predict(features, name)
            results[name] = {"prediction": pred, "confidence": conf}
            predictions.append(pred)
            confidences.append(conf)
        
        final_prediction = max(set(predictions), key=predictions.count)
        avg_confidence = np.mean(confidences)
        
        # Determine the "best" model based on confidence for this specific prediction
        best_model = max(results.items(), key=lambda x: x[1]["confidence"])[0]
        
        return {
            "final_prediction": final_prediction,
            "avg_confidence": avg_confidence,
            "best_model": best_model,
            "models": results
        }
    
    def save_models(self):
        """Save trained models to disk."""
        if not self.is_trained:
            logger.warning("No trained models to save.")
            return
        
        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f"{name}_model.joblib")
            joblib.dump(model, path)
            logger.info(f"Saved {name} to {path}")
        
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        features_path = os.path.join(self.model_dir, "feature_names.joblib")
        joblib.dump(self.feature_names, features_path)
        
        logger.success("All models saved successfully.")
    
    def load_models(self):
        """Load trained models from disk."""
        try:
            for name in ["random_forest", "svm", "xgboost"]:
                path = os.path.join(self.model_dir, f"{name}_model.joblib")
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
            
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            features_path = os.path.join(self.model_dir, "feature_names.joblib")
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            
            self.is_trained = len(self.models) > 0
            logger.success("Models loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            return False
    
    def get_feature_importance(self, model_name="random_forest") -> dict:
        """Get feature importance from tree-based models."""
        if model_name not in ["random_forest", "xgboost"]:
            return {}
        
        model = self.models.get(model_name)
        if model is None or self.feature_names is None:
            return {}
        
        importance = model.feature_importances_
        return dict(zip(self.feature_names, importance))


def train_and_evaluate():
    """Train and evaluate all classification models."""
    from iot.simulator import IoTSimulator
    
    logger.info("Generating training data...")
    simulator = IoTSimulator()
    data = simulator.generate_dataset(n_samples=2000, include_intrusion=True)
    
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Label distribution:\n{data['label'].value_counts()}")
    
    classifier = IntrusionClassifier()
    results = classifier.train(data)
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Confusion Matrix: {result['confusion_matrix']}")
    
    classifier.save_models()
    
    print("\n" + "="*60)
    print("Feature Importance (Random Forest)")
    print("="*60)
    importance = classifier.get_feature_importance("random_forest")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feature}: {imp:.4f}")
    
    return classifier


if __name__ == "__main__":
    train_and_evaluate()
