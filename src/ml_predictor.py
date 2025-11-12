"""
Machine Learning models for arbitrage opportunity prediction.
Uses various ML algorithms to predict profitable arbitrage opportunities.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import os
from datetime import datetime, timedelta
from collections import deque
import json


class ArbitragePredictor:
    """Machine learning predictor for arbitrage opportunities."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # ML Models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        }

        # Feature scaler
        self.scaler = StandardScaler()

        # Historical data for training
        self.feature_history = deque(maxlen=10000)
        self.target_history = deque(maxlen=10000)

        # Model performance tracking
        self.model_performance = {}

        # Load existing models if available
        self._load_models()

    def extract_features(self, opportunity_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from arbitrage opportunity data for ML prediction.

        Args:
            opportunity_data: Dictionary containing opportunity information

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Basic arbitrage features
        features.append(opportunity_data.get('profit_percentage', 0.0))
        features.append(opportunity_data.get('profit_usd', 0.0))
        features.append(opportunity_data.get('volatility', 0.0))
        features.append(opportunity_data.get('liquidity_score', 0.0))

        # Exchange features (one-hot encoded)
        exchange = opportunity_data.get('exchange', 'binance')
        exchanges = ['binance', 'coinbase', 'kraken', 'kucoin', 'okx']
        for ex in exchanges:
            features.append(1.0 if exchange == ex else 0.0)

        # Currency pair features
        base_currency = opportunity_data.get('base_currency', 'ETH')
        quote_currency = opportunity_data.get('quote_currency', 'BTC')
        alt_currency = opportunity_data.get('alt_currency', '')

        # Currency volatility (simplified)
        currency_volatility = {
            'BTC': 0.03, 'ETH': 0.04, 'BNB': 0.05, 'ADA': 0.06,
            'SOL': 0.07, 'DOT': 0.06, 'AVAX': 0.08, 'MATIC': 0.07
        }

        features.append(currency_volatility.get(base_currency, 0.05))
        features.append(currency_volatility.get(quote_currency, 0.03))
        features.append(currency_volatility.get(alt_currency, 0.05))

        # Time-based features
        timestamp = opportunity_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        features.append(timestamp.hour / 24.0)  # Hour of day (normalized)
        features.append(timestamp.weekday() / 7.0)  # Day of week (normalized)

        # Orderbook depth features
        orderbook_depth = opportunity_data.get('orderbook_depth', 2)
        features.append(orderbook_depth)

        # Historical performance features (last 10 similar opportunities)
        similar_opportunities = self._get_similar_opportunities(opportunity_data)
        if similar_opportunities:
            success_rate = np.mean([1 if opp.get('profit_percentage', 0) > 0 else 0
                                  for opp in similar_opportunities])
            avg_profit = np.mean([opp.get('profit_percentage', 0) for opp in similar_opportunities])
            features.extend([success_rate, avg_profit])
        else:
            features.extend([0.5, 0.0])  # Default values

        # Market condition features
        features.append(self._get_market_volatility())
        features.append(self._get_market_trend())

        return np.array(features)

    def _get_similar_opportunities(self, opportunity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get historically similar opportunities."""
        similar = []
        base = opportunity_data.get('base_currency')
        quote = opportunity_data.get('quote_currency')
        direction = opportunity_data.get('direction')

        for i, features in enumerate(self.feature_history):
            if i >= len(self.target_history):
                continue

            # Simple similarity check (same currencies and direction)
            if (len(features) > 1 and
                features[1] == opportunity_data.get('profit_percentage', 0) and
                direction == 'forward'):  # Simplified check
                similar.append({
                    'profit_percentage': self.target_history[i],
                    'features': features
                })

            if len(similar) >= 10:
                break

        return similar

    def _get_market_volatility(self) -> float:
        """Get current market volatility (simplified implementation)."""
        # In a real implementation, this would analyze recent price movements
        return 0.05  # Placeholder

    def _get_market_trend(self) -> float:
        """Get current market trend (-1 to 1, negative = bearish, positive = bullish)."""
        # In a real implementation, this would use technical indicators
        return 0.1  # Slightly bullish placeholder

    def add_training_example(self, opportunity_data: Dict[str, Any], outcome: bool):
        """
        Add a training example to the dataset.

        Args:
            opportunity_data: Feature data for the opportunity
            outcome: True if profitable, False otherwise
        """
        features = self.extract_features(opportunity_data)
        self.feature_history.append(features)
        self.target_history.append(1 if outcome else 0)

    def train_models(self, test_size: float = 0.2):
        """
        Train all ML models on historical data.

        Args:
            test_size: Fraction of data to use for testing
        """
        if len(self.feature_history) < 100:
            self.logger.warning("Not enough training data. Need at least 100 examples.")
            return

        # Prepare data
        X = np.array(list(self.feature_history))
        y = np.array(list(self.target_history))

        if len(np.unique(y)) < 2:
            self.logger.warning("Need both positive and negative examples for training.")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train each model
        for name, model in self.models.items():
            try:
                self.logger.info(f"Training {name} model...")

                # Train model
                model.fit(X_train_scaled, y_train)

                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                }

                self.model_performance[name] = metrics

                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()

                self.logger.info(f"{name} performance: {metrics}")

            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")

        # Save trained models
        self._save_models()

    def predict_opportunity(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict whether an arbitrage opportunity will be profitable.

        Args:
            opportunity_data: Opportunity data for prediction

        Returns:
            Dictionary with predictions from all models
        """
        features = self.extract_features(opportunity_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        predictions = {}

        for name, model in self.models.items():
            try:
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0][1]

                predictions[name] = {
                    'prediction': bool(prediction),
                    'probability': float(probability),
                    'confidence': abs(probability - 0.5) * 2  # Scale to 0-1
                }
            except Exception as e:
                self.logger.error(f"Error predicting with {name}: {e}")
                predictions[name] = {
                    'prediction': False,
                    'probability': 0.0,
                    'confidence': 0.0
                }

        # Ensemble prediction (majority vote)
        positive_votes = sum(1 for pred in predictions.values() if pred['prediction'])
        ensemble_prediction = positive_votes > len(predictions) / 2
        ensemble_probability = np.mean([pred['probability'] for pred in predictions.values()])

        predictions['ensemble'] = {
            'prediction': ensemble_prediction,
            'probability': ensemble_probability,
            'confidence': abs(ensemble_probability - 0.5) * 2
        }

        return predictions

    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        return self.model_performance.copy()

    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance for a specific model."""
        if model_name not in self.models:
            return {}

        model = self.models[model_name]

        if not hasattr(model, 'feature_importances_'):
            return {}

        # Feature names (simplified)
        feature_names = [
            'profit_percentage', 'profit_usd', 'volatility', 'liquidity_score',
            'is_binance', 'is_coinbase', 'is_kraken', 'is_kucoin', 'is_okx',
            'base_volatility', 'quote_volatility', 'alt_volatility',
            'hour_of_day', 'day_of_week', 'orderbook_depth',
            'historical_success_rate', 'historical_avg_profit',
            'market_volatility', 'market_trend'
        ]

        importance_dict = {}
        for name, importance in zip(feature_names, model.feature_importances_):
            importance_dict[name] = float(importance)

        return importance_dict

    def _save_models(self):
        """Save trained models to disk."""
        try:
            for name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            # Save scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save performance metrics
            perf_path = os.path.join(self.model_dir, "performance.json")
            with open(perf_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)

            self.logger.info("Models saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def _load_models(self):
        """Load trained models from disk."""
        try:
            for name in self.models.keys():
                model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)

            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

            # Load performance metrics
            perf_path = os.path.join(self.model_dir, "performance.json")
            if os.path.exists(perf_path):
                with open(perf_path, 'r') as f:
                    self.model_performance = json.load(f)

            self.logger.info("Models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def update_market_data(self, market_data: Dict[str, Any]):
        """Update internal market data for feature engineering."""
        # This would be called periodically to update market conditions
        # Implementation would depend on the specific market data structure
        pass

    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about the training data."""
        return {
            'total_examples': len(self.feature_history),
            'positive_examples': sum(self.target_history) if self.target_history else 0,
            'negative_examples': len(self.target_history) - sum(self.target_history) if self.target_history else 0,
            'class_balance': sum(self.target_history) / len(self.target_history) if self.target_history else 0,
            'models_trained': len([m for m in self.models.keys() if hasattr(self.models[m], 'n_features_in_')])
        }


class NeuralNetworkPredictor:
    """Neural network-based predictor for complex arbitrage patterns."""

    def __init__(self, input_dim: int = 20, hidden_dims: List[int] = [64, 32]):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            self.torch_available = True
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Define neural network architecture
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())

            self.model = nn.Sequential(*layers).to(self.device)
            self.criterion = nn.BCELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        except ImportError:
            self.torch_available = False
            self.logger.warning("PyTorch not available. Neural network predictor disabled.")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the neural network."""
        if not self.torch_available:
            return

        import torch

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device).unsqueeze(1)

        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the neural network."""
        if not self.torch_available:
            return np.zeros(len(X))

        import torch

        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.torch_available:
            return np.zeros(len(X))

        import torch

        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy().flatten()

        return outputs


class EnsemblePredictor:
    """Ensemble predictor combining multiple ML models."""

    def __init__(self):
        self.predictors = {
            'traditional': ArbitragePredictor(),
            'neural_net': NeuralNetworkPredictor()
        }
        self.weights = {
            'traditional': 0.7,
            'neural_net': 0.3
        }

    def train_all(self, X: np.ndarray, y: np.ndarray):
        """Train all predictors."""
        self.predictors['traditional'].train_models()

        if self.predictors['neural_net'].torch_available:
            self.predictors['neural_net'].train(X, y)

    def predict_ensemble(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble prediction."""
        # Get predictions from all models
        predictions = {}

        # Traditional ML models
        traditional_pred = self.predictors['traditional'].predict_opportunity(opportunity_data)
        predictions['traditional_ml'] = traditional_pred['ensemble']

        # Neural network (if available)
        if self.predictors['neural_net'].torch_available:
            features = self.predictors['traditional'].extract_features(opportunity_data)
            nn_pred = self.predictors['neural_net'].predict_proba(features.reshape(1, -1))[0]
            predictions['neural_network'] = {
                'prediction': nn_pred > 0.5,
                'probability': nn_pred,
                'confidence': abs(nn_pred - 0.5) * 2
            }

        # Weighted ensemble
        probabilities = [pred['probability'] for pred in predictions.values()]
        weights = list(self.weights.values())[:len(probabilities)]

        ensemble_prob = np.average(probabilities, weights=weights)
        ensemble_pred = ensemble_prob > 0.5

        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_prob,
            'individual_predictions': predictions,
            'confidence': abs(ensemble_prob - 0.5) * 2
        }
