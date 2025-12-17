"""
Prediction service using trained LSTM model.

Loads the trained model and provides predictions for stock prices.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.preprocessing import MinMaxScaler

from app.layer1_data_processing.market_data import fetch_market_data_sync


class StockLSTM(nn.Module):
    """LSTM model for stock price prediction (same architecture as training)."""
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


class PredictionService:
    """
    Service for making stock price predictions using the trained LSTM model.
    """
    
    def __init__(self, model_path: str = "./models/lstm_final.pt"):
        self.model_path = model_path
        self.model: Optional[StockLSTM] = None
        self.config: Dict = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from file."""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}")
            return
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                self.config = checkpoint['config']
                state_dict = checkpoint['model_state_dict']
            else:
                # Fallback for simpler checkpoint
                self.config = {
                    'seq_length': 30,
                    'hidden_size': 64,
                    'features': ['open', 'high', 'low', 'close', 'volume']
                }
                state_dict = checkpoint
            
            self.model = StockLSTM(
                input_size=len(self.config.get('features', ['open', 'high', 'low', 'close', 'volume'])),
                hidden_size=self.config.get('hidden_size', 64),
                num_layers=2,
                output_size=1
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, symbol: str) -> Dict:
        """
        Make a prediction for a given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., RELIANCE.NS)
        
        Returns:
            Dictionary with prediction details
        """
        if self.model is None:
            return {
                "symbol": symbol,
                "prediction": None,
                "error": "Model not loaded"
            }
        
        try:
            # Get recent data
            seq_length = self.config.get('seq_length', 30)
            features = self.config.get('features', ['open', 'high', 'low', 'close', 'volume'])
            
            # Fetch data (need extra days for sequence)
            df = fetch_market_data_sync(symbol, period="3mo", interval="1d")
            
            if df is None or len(df) < seq_length:
                return {
                    "symbol": symbol,
                    "prediction": None,
                    "error": "Insufficient data"
                }
            
            # Prepare columns
            df.columns = [col.lower() for col in df.columns]
            
            # Scale data
            scaler = MinMaxScaler()
            data = df[features].values[-seq_length:]
            scaled_data = scaler.fit_transform(df[features].values)[-seq_length:]
            
            # Create sequence
            sequence = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred_scaled = self.model(sequence).cpu().numpy()[0][0]
            
            # Inverse transform (approximate - use close price column index)
            current_price = df['close'].iloc[-1]
            close_idx = features.index('close')
            
            # Scale back
            dummy = np.zeros((1, len(features)))
            dummy[0, close_idx] = pred_scaled
            pred_unscaled = scaler.inverse_transform(dummy)[0, close_idx]
            
            # Calculate change
            price_change = pred_unscaled - current_price
            change_pct = (price_change / current_price) * 100
            
            # Determine action and confidence
            if change_pct > 1.0:
                action = "BUY"
                confidence = min(0.5 + change_pct / 10, 0.95)
            elif change_pct < -1.0:
                action = "SELL"
                confidence = min(0.5 + abs(change_pct) / 10, 0.95)
            else:
                action = "HOLD"
                confidence = 0.5 + (0.5 - abs(change_pct) / 2) * 0.3
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "predicted_price": float(pred_unscaled),
                "price_change": float(price_change),
                "change_pct": float(change_pct),
                "action": action,
                "confidence": float(confidence),
                "model": "LSTM",
            }
            
        except Exception as e:
            return {
                "symbol": symbol,
                "prediction": None,
                "error": str(e)
            }
    
    def predict_batch(self, symbols: List[str]) -> List[Dict]:
        """Predict for multiple symbols."""
        return [self.predict(symbol) for symbol in symbols]


# Global service instance
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get or create the prediction service singleton."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
