# 1. INTRODUCTION

## 1.1. Background of the Problem

Financial markets are high-dimensional, stochastic systems influenced by macroeconomic signals, institutional activity, liquidity cycles, and behavioral factors. In such environments, manual trading is often susceptible to delayed reactions, cognitive bias, and inconsistent risk discipline. Algorithmic trading emerged as a solution to these limitations by formalizing strategy logic into programmable decision processes.

Traditional algorithmic methods, however, are commonly constrained by static rules and limited adaptability under regime shifts. Recent advances in machine learning, particularly deep sequential models (LSTM) and reinforcement learning (PPO), have enabled richer market representation and adaptive policy learning. Yet, many deployed systems still operate with one-size-fits-all outputs, overlooking individual trader characteristics and risk appetite.

The Indian equity market presents unique challenges: NSE trading hours (9:15-15:30 IST), high retail participation, diverse volatility patterns across 52 NIFTY 50 constituent stocks, and the prevalence of both intraday and delivery-based trading. A meaningful next step is therefore not only to improve predictive capability but to integrate it with risk-aware action selection and user personalization in a complete engineering pipeline that targets these specific market characteristics.

## 1.2. Literature Survey

Recent literature highlights three important directions in algorithmic trading research:

**Deep Reinforcement Learning for Trading:** Huang et al. (2024) proposed a deep reinforcement learning framework with BiLSTM-Attention networks for algorithmic trading, demonstrating improved cumulative return and risk-adjusted performance over baseline trading heuristics. Their work on the DJIA dataset showed that temporal attention mechanisms significantly enhance feature extraction from noisy market data.

**Probabilistic Forecasting:** Li et al. (2024) introduced DeepAR-Attention probabilistic prediction for stock price series, improving uncertainty representation by modeling forecast distributions rather than only point estimates. This approach is particularly relevant for risk-aware position sizing, though our project uses a more conventional LSTM point-estimate approach for simplicity.

**Systematic Reviews:** Bhuiyan et al. (2025) conducted a systematic review of deep learning for algorithmic trading, identifying strong capability for non-linear feature extraction but noting overfitting, interpretability, and deployment robustness as key open challenges. Their review of over 100 papers confirms that hybrid LSTM-RL architectures remain an active research frontier.

**PPO in Financial Domains:** Schulman et al. (2017) introduced Proximal Policy Optimization, which has become the preferred RL algorithm for financial applications due to its stable training via clipped surrogate objectives. Stable-Baselines3 (Raffin et al., 2021) provides a production-grade implementation used in our project.

These findings suggest that the most practical architecture is one that combines deep predictive modeling (LSTM), policy optimization (PPO), and explicit risk personalization while preserving operational reliability through fallback mechanisms and multiple deployment modes.

## 1.3. Problem Statement and Necessity

**Problem Statement:**

Despite significant research progress, many real-world trading tools still suffer from fragmented design:
- Prediction modules exist without coherent execution logic
- Decision modules operate without personalized risk context
- Frontend interfaces lack behavior-level feedback mechanisms
- Deployments show weak resilience to missing models or infrastructure constraints

Therefore, the core problem addressed in this project is the development of an integrated, user-aware, and deployment-ready platform that can:
1. Ingest and process market data for 52 NIFTY 50 stocks in near real-time
2. Generate model-assisted trading action signals using LSTM price prediction and PPO policy optimization
3. Adapt output behavior to individual trader profiles through a structured questionnaire and behavior vector
4. Expose reliable REST/WebSocket APIs and a responsive React-based UI for practical usage
5. Support multiple runtime modes for academic demonstration and production scalability

**Necessity:**

- **Integration Gap:** Most existing academic projects focus on either price prediction or RL-based trading, not both. Our pipeline combines LSTM forecasting with PPO decision-making, where the PPO action overrides the LSTM rule-based action for a unified signal.

- **Personalization Gap:** Standard trading systems treat all users identically. Our behavior profiling system maps 30+ questionnaire responses to a 15-dimensional normalized behavior vector that adjusts position sizing, risk parameters, and triggers per-user PPO model retraining.

- **Deployment Gap:** Many research prototypes are not designed for real-world use. Our three-tier storage strategy (in-memory for demo, Firestore for cloud, PostgreSQL for production) with automatic fallback ensures practical deployability.

- **Resilience Gap:** When ML models are unavailable (missing files, training failures), the system degrades gracefully to rule-based signals rather than crashing, ensuring continuous service for the user.

## 1.4. Motivation

The motivation for this project originates from the mismatch between academic model performance and practical trader usability. High predictive accuracy alone does not guarantee meaningful decision support unless recommendations are aligned with volatility tolerance, capital discipline, and behavioral consistency.

**Key Motivational Drivers:**

1. **Bridging Research and Practice:** By unifying deep sequential modeling (LSTM), policy-based reinforcement learning (PPO), and trader behavior modeling into a single pipeline, this work aims to bridge academic research capability with applied decision quality.

2. **Addressing Indian Market Specificity:** Most algorithmic trading research focuses on US markets (NYSE/NASDAQ). This project targets NSE/BSE with proper symbol normalization, market hours detection (9:15-15:30 IST), and support for 52 Indian stocks.

3. **Behavior-Aware AI:** Traditional systems ignore trader psychology. Our approach encodes trader preferences into a mathematical behavior vector that directly influences position sizing (via Kelly Criterion), risk scoring (0-1 scale), and even triggers personalized PPO model retraining.

4. **Complete Engineering Pipeline:** Rather than a standalone model, this project delivers a full-stack web application with auth, real-time WebSocket updates, backtesting, and trade evaluation — demonstrating the end-to-end engineering required for real-world deployment.

## 1.5. Feasibility

**Technical Feasibility:**

- **Mature Open-Source Ecosystem:** The project leverages FastAPI (Python 3.12) for async backend services, PyTorch for LSTM modeling, Stable-Baselines3 for PPO, Gymnasium for the trading environment, Next.js 16 with React 19 for the frontend, and Docker Compose for orchestration — all mature, well-documented frameworks.
- **Data Access:** yfinance provides free access to NSE/BSE historical and real-time data through the Yahoo Finance API, with proper `.NS` and `.BO` suffix handling.
- **Model Training:** LSTM training completes in ~5 minutes on CPU (23,167 samples, 30 epochs). PPO training takes ~2-5 minutes for 30,000 timesteps. Both models are lightweight (~39K parameters for LSTM, ~256x256 MLP for PPO).

**Operational Feasibility:**
- **Dual Runtime Modes:** Demo mode (in-memory storage, no database) enables zero-infrastructure demonstrations. Production mode (PostgreSQL + Redis) supports scalable deployment.
- **Fallback Behavior:** The system operates even without trained models, returning rule-based signals with appropriate confidence levels.
- **Environment-Driven Configuration:** All settings are controlled through `.env` variables with defaults that work out-of-the-box.

**Economic Feasibility:**
- **Zero Licensing Cost:** The entire software stack is open-source with no licensing fees.
- **Minimal Hardware Requirements:** Both training and inference run on standard laptops. No GPU required (though CUDA is supported if available).
- **Data Costs:** yfinance provides free market data; no paid API subscriptions are required for the prototype.
