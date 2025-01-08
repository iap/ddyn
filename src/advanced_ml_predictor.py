import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """Fit the anomaly detection model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        
    def predict(self, X):
        """Predict anomalies (-1 for anomalies, 1 for normal)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def score_samples(self, X):
        """Get anomaly scores for samples."""
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

class TimeSeriesCV:
    def __init__(self, n_splits=5):
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        
    def split(self, X):
        """Generate time series cross-validation splits."""
        return self.cv.split(X)
        
    def get_metrics(self, model, X, y):
        """Calculate cross-validation metrics."""
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, test_idx in self.cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics['accuracy'].append(model.score(X_test, y_test))
            metrics['precision'].append(precision_score(y_test, y_pred))
            metrics['recall'].append(recall_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred))
            
        return {k: np.mean(v) for k, v in metrics.items()}

class AdvancedIPPredictor:
    def __init__(self, model_dir='models/advanced'):
        self.model_dir = model_dir
        self.model_path = f"{model_dir}/ensemble_model.joblib"
        self.scaler_path = f"{model_dir}/scaler.joblib"
        self.anomaly_detector_path = f"{model_dir}/anomaly_detector.joblib"
        self.sequence_length = 24
        self.model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = AnomalyDetector()
        self.cv = TimeSeriesCV()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize or load existing models."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.anomaly_detector = joblib.load(self.anomaly_detector_path)
            else:
                self._create_new_models()
        except Exception as e:
            print(f"Error loading models: {e}")
            self._create_new_models()

    def _create_new_models(self):
        """Create new ensemble model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

    def _extract_advanced_features(self, df):
        """Extract sophisticated features from IP change data."""
        features = pd.DataFrame()
        
        # Time-based features
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['day_of_month'] = df['timestamp'].dt.day
        features['month'] = df['timestamp'].dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['hour_sin'] = np.sin(2 * np.pi * features['hour']/24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour']/24)
        
        # Set timestamp as index for rolling operations
        df_rolling = df.set_index('timestamp')
        
        # Rolling statistics with time-based windows
        windows = {
            '1h': 1,
            '6h': 6,
            '12h': 12,
            '24h': 24,
            '7d': 24*7
        }
        
        for window_name, hours in windows.items():
            window_size = hours
            rolled = df_rolling['ip_changed'].rolling(window=window_size, min_periods=1)
            features[f'changes_{window_name}'] = rolled.sum().values
            features[f'change_rate_{window_name}'] = rolled.mean().values
        
        # Time since last change
        features['time_since_last_change'] = df['timestamp'].diff().dt.total_seconds()
        features['time_since_last_change_hours'] = features['time_since_last_change'] / 3600
        
        # Fill NaN values with appropriate defaults
        features = features.fillna(0)
        
        return features

    def train(self, data_file='data/ip_changes.parquet'):
        """Train models with cross-validation and anomaly detection."""
        try:
            df = pd.read_parquet(data_file)
            if len(df) < self.sequence_length + 1:
                print("Insufficient data for training")
                return False

            # Extract features
            features = self._extract_advanced_features(df)
            target = df['ip_changed'].astype(int)

            # Train anomaly detector
            self.anomaly_detector.fit(features)
            
            # Detect anomalies
            is_anomaly, anomaly_probs = self.detect_anomalies(features)
            print(f"\nDetected {sum(is_anomaly)} anomalous patterns out of {len(features)} samples")
            
            # Remove anomalies for training main models
            normal_idx = ~is_anomaly
            features_clean = features[normal_idx]
            target_clean = target[normal_idx]

            # Scale features
            X_scaled = self.scaler.fit_transform(features_clean)
            
            # Train model
            print("\nTraining random forest model...")
            self.model.fit(X_scaled, target_clean)

            # Save models
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.anomaly_detector, self.anomaly_detector_path)

            # Perform cross-validation
            metrics = self.cv.get_metrics(self.model, X_scaled, target_clean)
            print("\nCross-validation Results:")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.2%}")

            return True

        except Exception as e:
            print(f"Error training models: {e}")
            return False

    def predict_change_probability(self, current_time=None):
        """Predict IP change probability with anomaly detection."""
        if current_time is None:
            current_time = datetime.now()

        try:
            df = pd.read_parquet('data/ip_changes.parquet')
            if len(df) < self.sequence_length:
                return 0.5, False
                
            df = df[df['timestamp'] <= current_time]
            df = df.sort_values('timestamp').tail(self.sequence_length)
            
            features = self._extract_advanced_features(df)
            
            # Check for anomalies
            is_anomaly, anomaly_prob = self.detect_anomalies(features)
            is_current_anomaly = is_anomaly[-1]
            current_anomaly_prob = anomaly_prob[-1]
            
            # Get prediction from model
            X_scaled = self.scaler.transform(features.iloc[[-1]])
            prob = self.model.predict_proba(X_scaled)[0][1]
            
            # Adjust probability based on anomaly detection
            if is_current_anomaly:
                # Increase probability if pattern is anomalous
                prob = max(prob, 0.8)
            
            confidence = min(1.0, len(df) / (self.sequence_length * 2))
            adjusted_prob = (prob * confidence) + (0.5 * (1 - confidence))
            
            return adjusted_prob, is_current_anomaly

        except Exception as e:
            print(f"Error predicting change probability: {e}")
            return 0.5, False

    def detect_anomalies(self, features, threshold=0.1):
        """Detect anomalous IP change patterns."""
        try:
            # Get anomaly scores
            anomaly_scores = self.anomaly_detector.score_samples(features)
            
            # Convert to probabilities (higher score = more normal)
            anomaly_probs = np.exp(anomaly_scores) / (1 + np.exp(anomaly_scores))
            
            # Classify as anomaly if probability is below threshold
            is_anomaly = anomaly_probs < threshold
            
            return is_anomaly, anomaly_probs
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return np.zeros(len(features), dtype=bool), np.ones(len(features))
