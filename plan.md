# Comprehensive Product Plan — Firebase Migration + Trader Behavior Intelligence

## 1. Goals

We are redesigning the product around five major changes:

1. Migrate authentication and database from local/demo Postgres patterns to Firebase.
2. Replace single `risk_tolerance` value with a **behavior array** (multi-factor trader profile).
3. Expand questionnaire depth (more than 6 questions, richer answer types beyond low/moderate/high).
4. Add a trade-by-trade **feedback loop** that enforces or nudges behavior-aligned execution.
5. Update end-to-end product flow:
   - Login
   - Risk/behavior assessment
   - Profile computation
   - Model conditioning/training using profile + historical data
   - Parameter suggestions (SL/TP/time window/size)
   - Dashboard actions (Buy / Sell / Hold Buy / Hold Sell / Remain Idle)

---

## 2. Target Architecture

## 2.1 Identity
- Use **Firebase Authentication** for user sign-up/sign-in/session.
- Backend verifies Firebase ID token on each protected API request.
- Remove dependency on local JWT issuance for auth (JWT may still exist for internal service tokens if needed, but not for user identity).

## 2.2 Database
- Use **Cloud Firestore** as primary app database.
- Collections:
  - `users`
  - `risk_assessments`
  - `behavior_profiles`
  - `trade_events`
  - `trade_evaluations`
  - `model_runs`
  - `recommendations`
  - `watchlist_actions`

## 2.3 Compute / Model Layer
- Keep FastAPI + ML stack (LSTM/PPO), but add profile-conditioned feature engineering.
- Train/refresh model with:
  - historical market data
  - user behavior vector
  - trade outcome + evaluation history

---

## 3. New Behavior Array (Core Schema)

Instead of one risk scalar, use:

```json
{
  "capital_per_trade_pct": 0.0,
  "tp_sl_ratio_preference": 0.0,
  "max_profit_close_pct": 0.0,
  "trade_frequency_window_score": 0.0,
  "avg_holding_time_winner_min": 0.0,
  "avg_holding_time_loser_min": 0.0,
  "post_loss_rest_min": 0.0,
  "drawdown_sensitivity": 0.0,
  "streak_risk_adjustment": 0.0,
  "intraday_var_limit": 0.0,
  "entry_slippage_tolerance_bps": 0.0,
  "time_of_day_bias_vector": [0.0, 0.0, 0.0, 0.0],
  "news_proximity_buffer_min": 0.0,
  "partial_tp_preference": 0.0,
  "breakeven_migration_trigger_pct": 0.0,
  "breakeven_migration_time_min": 0.0
}
```

Notes:
- Normalize to `[0,1]` where practical.
- Keep raw values and normalized values for explainability + model use.

---

## 4. Questionnaire Design (Questions to Ask User)

Below are required questions mapped to your requested topics and high-utility features.

## 4.1 Capital Allocation & Trade Sizing
1. What percentage of your total capital do you want to risk in one trade? (numeric %)
2. What is the hard maximum capital allocation allowed per trade? (numeric % cap)
3. During a losing streak, should per-trade capital be reduced automatically? (Yes/No + reduction %)

## 4.2 TP/SL Preferences
4. Preferred TP:SL ratio for normal volatility markets? (e.g., 1.5:1, 2:1, 3:1)
5. Preferred TP:SL ratio for high-volatility markets? (same format)
6. Should SL be fixed or volatility-adaptive (ATR-based)? (single choice)

## 4.3 Profit Capture Behavior
7. At what profit % do you usually close full position? (numeric %)
8. Do you prefer partial take-profit before full exit? (Never/Sometimes/Often/Always)
9. If partial TP is used, what split do you prefer? (e.g., 50-30-20 or custom)

## 4.4 Overtrading & Impulse
10. Maximum number of trades allowed per hour? (integer)
11. Maximum number of trades allowed per day? (integer)
12. Minimum cooldown after a losing trade before opening next trade? (minutes)
13. Minimum cooldown after a winning trade? (minutes)

## 4.5 Holding-Time Behavior
14. Typical holding time target for winning trades? (minutes/hours)
15. Typical max holding time for losing trades before forced review/exit? (minutes/hours)
16. Should system warn when losing trades exceed planned holding time? (Yes/No)

## 4.6 Drawdown & Streak Controls
17. Maximum tolerated account drawdown before strict risk mode activates? (%)
18. After N consecutive losses, should risk be reduced? (N + reduction %)
19. After N consecutive wins, should risk remain capped to avoid overconfidence? (N + cap %)

## 4.7 VaR & Portfolio Risk
20. Max intraday VaR limit as % of equity? (numeric %)
21. Max simultaneous correlated positions allowed? (integer)
22. Should new trades be blocked if VaR limit is breached? (Yes/No)

## 4.8 Execution Quality & Slippage
23. Maximum acceptable entry slippage (in bps or %)? (numeric)
24. If slippage exceeds threshold, should order be canceled or resized? (choice)

## 4.9 Time-of-Day Performance
25. Which sessions do you prefer? (multi-select: Asia / London / NY / Overlap)
26. In low-liquidity session, should system reduce position size automatically? (Yes/No + %)

## 4.10 News Proximity Handling
27. Minimum minutes before high-impact news when new entries are blocked? (minutes)
28. During news window, preferred behavior? (Flatten / Hedge / Widen SL / No change)

## 4.11 Breakeven Management
29. At what unrealized profit % should SL move to breakeven? (%)
30. Max time allowed before moving SL to breakeven once trigger is hit? (minutes)

## 4.12 Behavioral Confidence & Discipline
31. How strictly should system enforce your rules? (Advisory / Soft block / Hard block)
32. If behavior deviates, allow override with reason logging? (Yes/No)
33. Preferred review frequency for behavior recalibration? (weekly/biweekly/monthly)

---

## 5. Feedback Loop Mechanism (Trade Evaluation Engine)

For every trade:

1. Capture planned parameters:
   - planned entry, SL, TP, position size, time window
2. Capture actual execution:
   - actual entry, slippage, hold duration, partial exits, final PnL
3. Compare against behavior profile:
   - rule match / violation list / severity score
4. Compute compliance score:
   - `trade_compliance_score` in `[0,100]`
5. Trigger response by policy:
   - Advisory: warning only
   - Soft block: friction prompt + reason required
   - Hard block: deny trade unless admin override mode
6. Learn and adapt:
   - store evaluation in `trade_evaluations`
   - periodic profile recalibration
   - retrain/reweight policy model with compliance + trade outcome features (PnL)

Strict behavior mode:
- If user enables strict mode, order execution must satisfy capital %, TP/SL, and cooldown constraints before placement.

### 5.1 Profit/Loss as Explicit Learning Signal
- Every closed trade contributes:
  - `realized_pnl`
  - `realized_pnl_pct`
  - `risk_adjusted_return` (e.g., normalized by SL distance or VaR budget)
  - `max_adverse_excursion` / `max_favorable_excursion`
- Model training objective should combine:
  - profitability (maximize expected return)
  - drawdown control (penalize deep losses)
  - behavior compliance (penalize rule-breaking trades)
- Example composite training score (conceptual):
  - `score = a*(pnl_pct) - b*(drawdown_penalty) - c*(compliance_violations) - d*(slippage_cost)`
- This ensures the system does not chase raw win-rate only; it learns **profitable and disciplined** behavior.

---

## 6. Product Flow (Final UX/Execution Flow)

1. User logs in with Firebase Auth.
2. If first login (or reassessment due), show full behavior assessment questionnaire.
3. Backend computes behavior array + risk envelope.
4. System initializes/updates user-conditioned model context:
   - historical market data
   - past trade data
   - behavior vector
5. System outputs personalized guardrails and recommendations:
   - position size range
   - stop loss / take profit bands
   - preferred holding window
   - no-trade windows (news / overtrading / drawdown risk)
6. Dashboard displays symbol-wise action:
   - Buy
   - Sell
   - Hold Buy
   - Hold Sell
   - Remain Idle
7. Before order placement, rule compliance is evaluated.
8. After trade closes, evaluation + feedback loop updates behavior intelligence.

---

## 7. Backend Implementation Plan

## Phase A — Firebase Foundation
1. Add Firebase Admin SDK to backend.
2. Create auth dependency:
   - verify Bearer Firebase ID token
   - map token UID to user profile document
3. Replace demo/local auth checks in protected routes.
4. Add Firestore repository layer (services for collections listed above).

## Phase B — Data Model Migration
1. Define Pydantic schemas for:
   - questionnaire responses
   - behavior array
   - trade evaluation report
2. Migrate profile endpoints:
   - `POST /profile/risk-assessment` -> returns behavior profile instead of single scalar.
3. Add endpoints:
   - `GET /profile/behavior`
   - `PUT /profile/behavior`
   - `POST /trades/evaluate`
   - `GET /trades/evaluations`

## Phase C — Decision Engine Integration
1. Extend state builder to append behavior features.
2. Add policy constraints module:
   - pre-trade gating (strict/soft/advisory)
3. Update signal generation output to include:
   - suggested size
   - suggested SL/TP
   - confidence + compliance notes

## Phase D — Feedback Learning
1. Build compliance score function.
2. Add periodic recalibration job:
   - adjust behavior vector from observed discipline/outcomes
3. Add retraining trigger rules (time-based or performance-based).
4. Include per-trade PnL and risk-adjusted outcome labels in each retrain dataset.
5. Add guardrail metrics for model promotion:
   - positive net return
   - max drawdown within threshold
   - profit factor above minimum threshold

---

## 8. Frontend Implementation Plan

## Phase E — Auth + Onboarding Flow
1. Replace current auth store logic with Firebase client SDK.
2. Maintain cookie/session bridge for Next middleware route protection.
3. Post-login gating:
   - if no assessment -> redirect to questionnaire wizard.

## Phase F — Questionnaire Experience
1. Multi-step dynamic form with numeric, slider, single/multi-select inputs.
2. Group by:
   - capital/risk
   - execution behavior
   - overtrading discipline
   - news/session preferences
3. Submit to backend and show resulting behavior profile summary.

## Phase G — Dashboard Intelligence
1. Add recommendation cards:
   - position size, SL/TP, time window, news risk state
2. Add action table per symbol:
   - Buy / Sell / Hold Buy / Hold Sell / Remain Idle
3. Show compliance indicator before execution.
4. Show post-trade evaluation feed and behavior score trend.

---

## 9. API Contract Changes (High-Level)

- Keep existing route prefixes where possible.
- Replace or augment payloads to include behavior array fields.
- All protected routes require Firebase token.

Examples:
- `POST /api/v1/profile/risk-assessment`
  - Input: detailed questionnaire answers
  - Output: `behavior_profile`, `risk_envelope`, `enforcement_mode`

- `POST /api/v1/trading/signals/{symbol}`
  - Output extended with:
    - `suggested_position_size_pct`
    - `suggested_sl_pct`
    - `suggested_tp_pct`
    - `time_window_hint`
    - `compliance_precheck`

---

## 10. Rollout Strategy

1. Feature-flag Firebase path while keeping current demo mode fallback.
2. Introduce behavior array in parallel with old scalar risk, then deprecate scalar.
3. Release questionnaire + profile first.
4. Release compliance engine second.
5. Release model conditioning/retraining integration third.
6. Monitor:
   - onboarding completion
   - compliance score trend
   - trade outcome stability
   - user override frequency
   - net PnL trend and rolling profit factor

### Suggested Retraining Cadence
- Light refresh: when `>=100` new closed trades **or** every 7 days.
- Full retrain: when `>=300` new closed trades **or** monthly.
- No model promotion if validation profitability/risk gates are not met.

---

## 11. Acceptance Criteria

1. User can authenticate via Firebase and access protected pages.
2. User completes expanded questionnaire and gets behavior profile.
3. System stores profile + trade evaluations in Firestore.
4. Signal responses include behavior-aware recommendations.
5. Trade placement is evaluated against user behavior rules.
6. Dashboard shows final action states:
   - Buy / Sell / Hold Buy / Hold Sell / Remain Idle.
7. Feedback loop updates recommendations based on observed behavior.

---

## 12. Immediate Next Build Steps

1. Add Firebase config + auth verification in backend.
2. Implement Firestore repository and migrate profile storage.
3. Build questionnaire schema (33 questions above) and API endpoint.
4. Replace frontend auth flow with Firebase login + token handling.
5. Integrate behavior profile into signal response and dashboard cards.
6. Implement trade evaluation endpoint with compliance scoring.
