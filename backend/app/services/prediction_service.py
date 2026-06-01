"""
Prediction service using trained LSTM model.

Loads the trained model and provides predictions for stock prices.
"""
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.preprocessing import MinMaxScaler

from app.layer1_data_processing.market_data import fetch_market_data_sync
from app.layer2_decision.action_space import ACTION_LABELS


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
    
    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _build_effective_behavior(
        self,
        behavior_array: Optional[Dict[str, float]],
        lstm_context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Collapse full behavior vector into PPO-facing knobs while preserving all question weights.
        """
        base = behavior_array or {}
        defaults = {
            "capital_per_trade_pct": 0.10,
            "tp_sl_ratio_preference": 0.25,  # normalized ratio (0..1)
            "max_profit_close_pct": 0.20,
            "trade_frequency_window_score": 0.15,
            "avg_holding_time_score": 0.20,
            "post_loss_rest_min": 0.10,
            "drawdown_sensitivity": 0.15,
            "streak_risk_adjustment": 0.25,
            "intraday_var_limit": 0.10,
            "entry_slippage_tolerance_bps": 0.10,
            "time_of_day_performance_bias": 0.50,
            "news_proximity_buffer_min": 0.10,
            "partial_tp_preference": 0.50,
            "breakeven_migration_trigger_pct": 0.10,
            "breakeven_migration_time_min": 0.20,
        }

        v = {k: float(base.get(k, d)) for k, d in defaults.items()}
        for k in v:
            v[k] = self._clamp(v[k], 0.0, 1.0)

        risk_pressure = np.mean([
            v["drawdown_sensitivity"],
            v["intraday_var_limit"],
            v["streak_risk_adjustment"],
            v["news_proximity_buffer_min"],
        ])
        aggression_pressure = np.mean([
            v["trade_frequency_window_score"],
            v["entry_slippage_tolerance_bps"],
            v["time_of_day_performance_bias"],
            v["partial_tp_preference"],
        ])
        discipline_pressure = np.mean([
            v["post_loss_rest_min"],
            v["breakeven_migration_trigger_pct"],
            v["breakeven_migration_time_min"],
            v["avg_holding_time_score"],
        ])

        effective = v.copy()
        effective["capital_per_trade_pct"] = self._clamp(
            v["capital_per_trade_pct"] * (1 - 0.35 * risk_pressure) * (1 + 0.20 * aggression_pressure),
            0.02,
            0.35,
        )
        effective["tp_sl_ratio_preference"] = self._clamp(
            v["tp_sl_ratio_preference"] * (1 + 0.20 * discipline_pressure) + 0.10 * aggression_pressure,
            0.05,
            1.0,
        )
        effective["max_profit_close_pct"] = self._clamp(
            v["max_profit_close_pct"] * (1 + 0.25 * discipline_pressure),
            0.05,
            0.60,
        )
        effective["post_loss_rest_min"] = self._clamp(
            v["post_loss_rest_min"] * (1 + 0.35 * risk_pressure),
            0.0,
            1.0,
        )
        effective["drawdown_sensitivity"] = self._clamp(
            v["drawdown_sensitivity"] * (1 + 0.25 * discipline_pressure),
            0.0,
            1.0,
        )
        # LSTM confidence + directional forecast influence PPO behavior inputs.
        if lstm_context:
            lstm_conf = self._clamp(float(lstm_context.get("confidence", 0.5)), 0.0, 1.0)
            lstm_change = float(lstm_context.get("change_pct", 0.0))
            directional_strength = self._clamp(abs(lstm_change) / 3.0, 0.0, 1.0) * lstm_conf
            direction_sign = 1.0 if lstm_change >= 0 else -1.0

            effective["capital_per_trade_pct"] = self._clamp(
                effective["capital_per_trade_pct"] * (0.75 + 0.70 * directional_strength),
                0.02,
                0.35,
            )
            effective["tp_sl_ratio_preference"] = self._clamp(
                effective["tp_sl_ratio_preference"] + (0.12 * directional_strength * direction_sign),
                0.05,
                1.0,
            )
            effective["max_profit_close_pct"] = self._clamp(
                effective["max_profit_close_pct"] * (0.85 + 0.60 * directional_strength),
                0.05,
                0.60,
            )
        return effective

    def _build_trade_plan(
        self,
        behavior_array: Dict[str, float],
        current_price: float,
        action: str,
        lstm_confidence: float,
        lstm_change_pct: float,
        ppo_confidence: float,
        atr_pct: float,
    ) -> Dict[str, float]:
        base_capital_pct = self._clamp(float(behavior_array.get("capital_per_trade_pct", 0.10)) * 100.0, 2.0, 35.0)
        tp_sl_ratio_norm = self._clamp(float(behavior_array.get("tp_sl_ratio_preference", 0.25)), 0.05, 1.0)
        base_tp_sl_ratio = self._clamp(tp_sl_ratio_norm * 8.0, 0.5, 8.0)
        base_profit_pct = self._clamp(float(behavior_array.get("max_profit_close_pct", 0.20)) * 100.0, 0.5, 20.0)

        directional_strength = self._clamp(abs(lstm_change_pct) / 3.0, 0.0, 1.0)
        combined_conf = self._clamp((0.6 * lstm_confidence) + (0.4 * ppo_confidence), 0.0, 1.0)
        volatility_boost = self._clamp(atr_pct / 4.0, 0.0, 1.0)

        capital_pct = self._clamp(
            base_capital_pct * (0.70 + 0.65 * combined_conf) * (0.80 + 0.40 * directional_strength),
            2.0,
            35.0,
        )
        tp_sl_ratio = self._clamp(
            base_tp_sl_ratio * (0.85 + 0.50 * combined_conf + 0.20 * directional_strength),
            0.5,
            8.0,
        )
        profit_target_pct = self._clamp(
            base_profit_pct * (0.70 + 0.70 * combined_conf) + (0.60 * volatility_boost),
            0.5,
            15.0,
        )

        stop_loss_pct = self._clamp(profit_target_pct / max(tp_sl_ratio, 1.0), 0.3, 5.0)
        if action in ("SELL", "HOLD SELL"):
            profit_target_exit_price = current_price * (1 - profit_target_pct / 100.0)
            stop_loss_price = current_price * (1 + stop_loss_pct / 100.0)
        else:
            profit_target_exit_price = current_price * (1 + profit_target_pct / 100.0)
            stop_loss_price = current_price * (1 - stop_loss_pct / 100.0)

        notional_capital = 100000.0
        capital_amount_inr = notional_capital * (capital_pct / 100.0)
        qty = int(capital_amount_inr / current_price) if current_price > 0 else 0

        return {
            "capital_per_trade_pct": round(capital_pct, 2),
            "tp_sl_ratio_target": round(tp_sl_ratio, 2),
            "capital_amount_inr": round(capital_amount_inr, 2),
            "position_qty_est": qty,
            "profit_target_pct": round(profit_target_pct, 2),
            "profit_target_exit_price": round(profit_target_exit_price, 2),
            "stop_loss_price": round(stop_loss_price, 2),
        }

    def _load_symbol_fallback_data(self, symbol: str, rows: int = 90) -> Optional[pd.DataFrame]:
        try:
            data_file = Path("./data/training_data.csv")
            if not data_file.exists():
                return None
            df = pd.read_csv(data_file)
            if "symbol" not in df.columns:
                return None
            sym = str(symbol).upper()
            base_symbol = sym.replace(".NS", "").replace(".BO", "")
            candidates = {sym, f"{base_symbol}.NS", f"{base_symbol}.BO", base_symbol}
            sdf = df[df["symbol"].astype(str).str.upper().isin(candidates)].copy()
            if sdf.empty:
                return None
            if "date" in sdf.columns:
                sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
                sdf = sdf.sort_values("date")
            sdf = sdf.tail(rows)
            rename_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            for c in ["open", "high", "low", "close", "volume"]:
                if c not in sdf.columns:
                    return None
            return sdf.rename(columns=rename_map)[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            return None

    def _resolve_ppo_model_path(self, user_id: Optional[str]) -> str:
        model_root = "./models"
        try:
            from app.config import settings
            model_root = settings.MODEL_PATH
        except Exception:
            # Keep local default if settings parsing fails.
            model_root = "./models"

        if user_id:
            user_model = os.path.join(model_root, "users", str(user_id), "ppo_trading_final.zip")
            if os.path.exists(user_model):
                return user_model
        return os.path.join(model_root, "ppo_trading_final.zip")

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
    
    def get_ppo_signal(
        self,
        symbol: str,
        risk_tolerance: float = 0.5,
        behavior_array: Optional[Dict[str, float]] = None,
        user_id: Optional[str] = None,
        lstm_context: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Get trading signal using the PPO agent.
        
        This uses the same environment logic as training to ensure consistency.
        """
        try:
            # 1. Get recent market data
            from app.layer1_data_processing.market_data import fetch_market_data_sync
            df = fetch_market_data_sync(symbol, period="1mo", interval="1d")
            if df is None or len(df) < 30:
                df = self._load_symbol_fallback_data(symbol, rows=120)
            
            if df is None or len(df) < 30:
                return {"action": "IDLE", "confidence": 0.5, "error": "Insufficient data for PPO"}
            
            # 2. Initialize a temporary environment to get the state
            from app.layer2_decision.trading_env import TradingEnv
            effective_behavior = self._build_effective_behavior(behavior_array, lstm_context=lstm_context)
            env = TradingEnv(df=df, risk_tolerance=risk_tolerance, behavior_array=effective_behavior)
            obs, _ = env.reset()
            current_price = float(df["Close"].iloc[-1])
            atr_pct = 0.0
            try:
                high = df["High"].astype(float)
                low = df["Low"].astype(float)
                close = df["Close"].astype(float)
                prev_close = close.shift(1)
                tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
                atr = tr.rolling(window=14, min_periods=2).mean().iloc[-1]
                atr_pct = float((atr / current_price) * 100.0) if current_price > 0 else 0.0
            except Exception:
                atr_pct = 0.0
            
            # 3. Load PPO model and predict
            from stable_baselines3 import PPO
            ppo_model_path = self._resolve_ppo_model_path(user_id)
            
            if not os.path.exists(ppo_model_path):
                return {"action": "IDLE", "confidence": 0.5, "error": "PPO model not found"}
            
            ppo_model = PPO.load(ppo_model_path)
            action, _states = ppo_model.predict(obs, deterministic=True)
            action_int = int(action)
            confidence = 0.8
            try:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=ppo_model.device).reshape(1, -1)
                dist = ppo_model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
                confidence = float(probs[action_int]) if len(probs) > action_int else 0.8
            except Exception:
                confidence = 0.8
            
            return {
                "action": ACTION_LABELS[action_int],
                "confidence": self._clamp(confidence, 0.0, 1.0),
                "model": "PPO",
                "behavior_effective": effective_behavior,
                "trade_plan": self._build_trade_plan(
                    effective_behavior,
                    current_price=current_price,
                    action=ACTION_LABELS[action_int],
                    lstm_confidence=float(lstm_context.get("confidence", 0.5)) if lstm_context else 0.5,
                    lstm_change_pct=float(lstm_context.get("change_pct", 0.0)) if lstm_context else 0.0,
                    ppo_confidence=self._clamp(confidence, 0.0, 1.0),
                    atr_pct=atr_pct,
                ),
            }
        except Exception as e:
            return {"action": "IDLE", "confidence": 0.5, "error": str(e)}

    def predict(
        self,
        symbol: str,
        risk_tolerance: float = 0.5,
        behavior_array: Optional[Dict[str, float]] = None,
        user_id: Optional[str] = None,
    ) -> Dict:
        """
        Make a prediction for a given stock symbol using both LSTM and PPO.
        """
        # Get LSTM Prediction
        lstm_result = self._get_lstm_prediction(symbol)
        
        # Get PPO Signal
        ppo_result = self.get_ppo_signal(
            symbol,
            risk_tolerance,
            behavior_array=behavior_array,
            user_id=user_id,
            lstm_context={
                "confidence": float(lstm_result.get("confidence", 0.5)),
                "change_pct": float(lstm_result.get("change_pct", 0.0)),
                "predicted_price": float(lstm_result.get("predicted_price", 0.0) or 0.0),
                "current_price": float(lstm_result.get("current_price", 0.0) or 0.0),
            } if lstm_result.get("error") is None else None,
        )
        
        # Merge results
        final_result = lstm_result.copy()
        if ppo_result.get("error") is None:
            # Prefer PPO action if available as it's the 'Decision Engine'
            final_result["action"] = ppo_result["action"]
            final_result["ppo_confidence"] = ppo_result["confidence"]
            final_result["confidence"] = ppo_result["confidence"]
            final_result["model"] = "LSTM+PPO"
            final_result["trade_plan"] = ppo_result.get("trade_plan")
            final_result["behavior_effective"] = ppo_result.get("behavior_effective")
        
        return final_result

    def _get_lstm_prediction(self, symbol: str) -> Dict:
        """Original LSTM prediction logic (internal)."""
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
            
            # Fetch data
            from app.layer1_data_processing.market_data import fetch_market_data_sync
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
            scaled_data = scaler.fit_transform(df[features].values)[-seq_length:]
            
            # Create sequence
            sequence = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred_scaled = self.model(sequence).cpu().numpy()[0][0]
            
            # Inverse transform
            current_price = df['close'].iloc[-1]
            close_idx = features.index('close')
            
            dummy = np.zeros((1, len(features)))
            dummy[0, close_idx] = pred_scaled
            pred_unscaled = scaler.inverse_transform(dummy)[0, close_idx]
            
            price_change = pred_unscaled - current_price
            change_pct = (price_change / current_price) * 100
            
            # Initial action based on LSTM
            if change_pct > 1.0:
                action = "BUY"
                confidence = min(0.5 + change_pct / 10, 0.95)
            elif change_pct < -1.0:
                action = "SELL"
                confidence = min(0.5 + abs(change_pct) / 10, 0.95)
            else:
                action = "IDLE"
                confidence = 0.5
            
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
