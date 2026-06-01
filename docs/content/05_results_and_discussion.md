# 5. RESULT AND DISCUSSION

## 5.1. Experimental and System Outcomes

**LSTM Price Predictor Results:**

| Metric | Value |
|--------|-------|
| Architecture | 2-layer LSTM, hidden=64, seq_len=30, 5 features (OHLCV) |
| Total Parameters | ~39,000 |
| Training Samples | 23,167 (52 stocks, 5 years, 80/20 split) |
| Best Validation Loss (MSE) | **0.000180** |
| Optimizer | Adam (lr=0.001) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Gradient Clipping | max_norm=1.0 |
| Training Time | ~5 minutes (CPU) |

The LSTM achieves a best validation MSE of 0.000180 (epoch 21), indicating strong predictive performance on normalized OHLCV sequences. The low loss is partially attributable to the per-symbol MinMaxScaler normalization, which constrains all values to [0,1] range. The final epoch validation loss was 0.000230.

**PPO Trading Agent Results:**

| Metric | Value |
|--------|-------|
| Algorithm | PPO (Stable-Baselines3) |
| Policy Network | MlpPolicy, net_arch=[256, 256] |
| Training Timesteps | 30,000 |
| Action Space | Discrete(5): HOLD_BUY, HOLD_SELL, BUY, SELL, IDLE |
| Observation Space | Box(34,) |
| Average Return (5 eval episodes) | **132.28%** |
| Sharpe Ratio (evaluation) | **0.66** |
| Transaction Cost | 0.1% per trade |
| Initial Capital | 100,000 |
| Training Time | ~2-5 minutes (CPU) |

The PPO agent achieves a 132.28% average return with a Sharpe ratio of 0.66 across 5 evaluation episodes. The Sharpe ratio above 0.5 indicates favorable risk-adjusted returns. The training completes in 2-5 minutes on CPU due to the lightweight network architecture (two hidden layers of 256 units).

**System Functionality Delivered:**

1. **Market Data and Technical Indicators:** Real-time data for 52 NIFTY 50 stocks via yfinance, 30+ technical indicators computed, OHLCV history (up to 180 data points).

2. **Trading Signals:** Combined LSTM+PPO signal generation. Action space: BUY, SELL, HOLD_BUY, HOLD_SELL, IDLE. Confidence scoring (LSTM: 0.5-0.95 based on change_pct, PPO: hardcoded 0.8). Graceful fallback to rule-based when models unavailable.

3. **WebSocket Live Prices:** `/api/v1/ws/prices` with subscribe/unsubscribe protocol, 30-second polling interval, simulated real-time updates.

4. **Behavior Profiling:** 30+ question assessment . 15-dim behavior vector (0-1 normalized) . risk category (Conservative/Moderate/Growth/Aggressive) . position sizing recommendation. Per-user PPO retraining triggered on assessment submission.

5. **Backtesting:** Historical simulation through TradingEnv with PPO (or random fallback). Metrics: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor.

6. **Full-Stack Delivery:** Authenticated Next.js dashboard with middleware route protection, Axios interceptors, Zustand state persistence, Recharts charting (price, equity, indicators), React Hook Form + Zod validation.

## 5.2. Risk Analysis

**Technical Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| LSTM inference scaling mismatch | Reduced prediction accuracy | Use per-symbol saved scalers instead of fresh fit-on-query |
| PPO confidence hardcoded at 0.8 | Misleading confidence display | Use actual action probabilities from policy network |
| State dimension mismatch (30 vs 34) | Unused zero-padding in state | Align state_builder output dimension with TradingEnv expectation |
| BacktestService path sensitivity | File-not-found errors | Use absolute path resolution from file location |
| No test coverage | Undetected regressions | Add pytest backend tests and Vitest frontend tests |

**Operational Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| yfinance rate limits | Data fetch failures | 5-minute caching, graceful error handling |
| WebSocket is simulated (30s poll) | Non-real-time prices | Document as limitation; upgrade to WebSocket data provider for production |
| Demo mode state resets on restart | Data loss in demo | Clear documentation; persistence mode available via Firebase/PostgreSQL |
| Docker Node version mismatch | Frontend build failure | Update Dockerfile from node:18-slim to node:20-slim |

**Model Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Train-inference skew for PPO | Poor generalization | Train with diverse behavior arrays; validate on held-out profiles |
| No model versioning | Cannot roll back | Add model registry with timestamped checkpoints |
| LSTM trained on combined multi-stock data | Missed stock-specific patterns | Evaluate per-stock vs pooled training approaches |
| PPO training on single environment | Limited experience diversity | Implement vectorized environments for future work |

## 5.3. Critical Discussion

**Strengths:**

1. **Integrated Pipeline:** Unlike most academic projects that focus on isolated components, this project delivers a complete pipeline from data ingestion through ML inference to user-facing dashboard.
2. **Three-Layer Architecture:** Clear separation of concerns between data processing, RL decision-making, and behavior personalization enables independent testing and future component swaps.
3. **Behavior Personalization:** The 15-dim behavior vector approach is novel in integrating trader psychology directly into the RL observation space and triggering personalized model retraining.
4. **Multiple Runtime Modes:** Supporting demo/Firebase/production modes with graceful fallback demonstrates production-aware engineering.
5. **Real Training Results:** The LSTM (best val loss 0.000180, final 0.000230) and PPO (132.28% return, 0.66 Sharpe) provide concrete evidence of model viability.

**Limitations:**

1. **Simulated Real-Time Data:** yfinance polling every 30 seconds is not a true live feed. Production deployment requires a real-time market data provider.
2. **No Actual Trade Execution:** The system generates signals but does not execute trades through a broker API (Zerodha, Upstox, etc.).
3. **Model Governance Gaps:** No versioning, experiment tracking (MLflow/W&B), or training data drift monitoring.
4. **Limited Backtesting Rigor:** No walk-forward analysis, Monte Carlo simulation, or benchmark comparison against buy-and-hold.
5. **No Test Coverage:** Both backend and frontend lack automated tests, which is a significant quality risk.
6. **DeepAR Unused:** The DeepAR probabilistic forecasting training script is implemented (`train_deepar.py`) but the model has not been integrated into the inference pipeline.

**Comparison with Related Work:**
- Huang et al. (2024) achieve strong results with BiLSTM-Attention but focus on prediction only, not end-to-end deployment.
- Li et al. (2024) use DeepAR-Attention for uncertainty quantification but do not incorporate RL decision-making.
- Bhuiyan et al. (2025) systematically review 100+ papers and note that integrated prediction-RL-personalization pipelines are rare — this project addresses exactly that gap.
- The 132.28% return and 0.66 Sharpe ratio are competitive with reported results, though direct comparison is limited by different market regimes and evaluation protocols.
