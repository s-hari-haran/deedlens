"""
DeedLens Regression
Predicts property values using extracted features.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class PredictionResult:
    """Result of value prediction."""
    predicted_value: float
    confidence: float
    prediction_range: Tuple[float, float]


class PropertyValuePredictor:
    """
    Predicts property values based on extracted features.
    Uses Random Forest or Gradient Boosting.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install it first.")
        
        self.model_type = model_type
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ["area", "bedrooms", "bathrooms", "year", "location_score"]
    
    def _create_model(self):
        """Create the ML model."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            return LinearRegression()
    
    def _extract_features(self, property_data: Dict) -> np.ndarray:
        """Extract features from property data."""
        features = [
            property_data.get("area", 0),
            property_data.get("bedrooms", 2),
            property_data.get("bathrooms", 1),
            property_data.get("year", 2020),
            property_data.get("location_score", 5),
        ]
        return np.array(features).reshape(1, -1)
    
    def train(
        self,
        properties: List[Dict],
        values: List[float],
        test_size: float = 0.2
    ) -> Dict:
        """
        Train the model on property data.
        
        Args:
            properties: List of property feature dictionaries
            values: List of property values
            test_size: Fraction for test set
        
        Returns:
            Dictionary with training metrics
        """
        # Extract features
        X = np.array([
            [
                p.get("area", 0),
                p.get("bedrooms", 2),
                p.get("bathrooms", 1),
                p.get("year", 2020),
                p.get("location_score", 5),
            ]
            for p in properties
        ])
        
        y = np.array(values)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        return metrics
    
    def predict(self, property_data: Dict) -> PredictionResult:
        """
        Predict property value.
        
        Args:
            property_data: Dictionary with property features
        
        Returns:
            PredictionResult with predicted value and confidence
        """
        if not self.is_trained:
            # Return a rough estimate based on area
            area = property_data.get("area", 1000)
            price_per_sqft = 10000  # Default assumption
            estimated = area * price_per_sqft
            
            return PredictionResult(
                predicted_value=estimated,
                confidence=0.3,
                prediction_range=(estimated * 0.7, estimated * 1.3)
            )
        
        # Extract and scale features
        features = self._extract_features(property_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # For Random Forest, we can get prediction intervals
        if self.model_type == "random_forest":
            predictions = np.array([
                tree.predict(features_scaled)[0]
                for tree in self.model.estimators_
            ])
            
            lower = np.percentile(predictions, 10)
            upper = np.percentile(predictions, 90)
            confidence = 1 - (upper - lower) / prediction if prediction > 0 else 0.5
        else:
            lower = prediction * 0.85
            upper = prediction * 1.15
            confidence = 0.7
        
        return PredictionResult(
            predicted_value=float(prediction),
            confidence=float(max(0, min(1, confidence))),
            prediction_range=(float(lower), float(upper))
        )
    
    def save(self, path: str):
        """Save the trained model."""
        import pickle
        
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        with open(model_path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        
        with open(model_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(model_path / "config.json", "w") as f:
            json.dump({
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "is_trained": self.is_trained
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "PropertyValuePredictor":
        """Load a trained model."""
        import pickle
        
        model_path = Path(path)
        
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        instance = cls(model_type=config["model_type"])
        
        with open(model_path / "model.pkl", "rb") as f:
            instance.model = pickle.load(f)
        
        with open(model_path / "scaler.pkl", "rb") as f:
            instance.scaler = pickle.load(f)
        
        instance.is_trained = config["is_trained"]
        
        return instance


def predict_property_value(property_data: Dict) -> Dict:
    """
    Convenience function to predict property value.
    
    Args:
        property_data: Dictionary with property features
    
    Returns:
        Dictionary with prediction results
    """
    predictor = PropertyValuePredictor()
    result = predictor.predict(property_data)
    
    return {
        "predicted_value": result.predicted_value,
        "confidence": result.confidence,
        "range_low": result.prediction_range[0],
        "range_high": result.prediction_range[1]
    }


if __name__ == "__main__":
    # Example usage
    property_data = {
        "area": 2400,
        "bedrooms": 3,
        "bathrooms": 2,
        "year": 2024,
        "location_score": 8
    }
    
    result = predict_property_value(property_data)
    
    print("=== Prediction ===")
    print(f"Predicted Value: Rs. {result['predicted_value']:,.0f}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Range: Rs. {result['range_low']:,.0f} - Rs. {result['range_high']:,.0f}")
