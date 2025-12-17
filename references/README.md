# References

This folder contains the research papers and presentations that form the theoretical foundation of this project.

## Research Papers

### 1. Huang et al. (2024)
**File**: `1-s2.0-S095741742303083X-main.pdf`

**Title**: "A novel deep reinforcement learning framework with BiLSTM-Attention networks for algorithmic trading"

**Journal**: Expert Systems With Applications, Volume 240

**Key Contributions**:
- BiLSTM-Attention architecture for feature extraction
- Deep Reinforcement Learning for trading decisions
- Modular two-stage approach (prediction → optimization)

**Our Implementation**: BiLSTM encoder in Layer 1, PPO agent architecture in Layer 2

---

### 2. Bhuiyan et al. (2025)
**File**: `1-s2.0-S2590005625000177-main.pdf`

**Title**: "Deep learning for algorithmic trading: A systematic review of predictive models and optimization strategies"

**Journal**: Array, Volume 26

**Key Contributions**:
- Comprehensive review of DL models for trading
- Classification of predictive vs optimization approaches
- Best practices for model design

**Our Implementation**: Architecture patterns, training methodologies

---

### 3. Li et al. (2024)
**File**: `s00521-024-09916-3.pdf`

**Title**: "DeepAR-Attention probabilistic prediction for stock price series"

**Journal**: Neural Computing and Applications, Volume 36

**Key Contributions**:
- DeepAR model with attention mechanism
- Probabilistic forecasting (μ, σ)
- Uncertainty quantification for risk management

**Our Implementation**: DeepAR-Attention model in Layer 1 for price prediction

---

## Project Presentation

### Major Project Presentation 1.pptx

**Contents**:
- Problem overview and purpose
- The trader's cognitive process
- Current DL models in finance
- Problem statement
- Research objectives
- Proposed two-stage solution
- Conclusion

**Key Points from Presentation**:
1. Two-stage architecture (Prediction → Optimization)
2. Integration of trader behavior (risk tolerance, timeframe)
3. Use of probabilistic forecasts
4. Optimization for Sharpe Ratio (risk-adjusted returns)

---

## Citation Format

If referencing these papers in documentation:

```bibtex
@article{huang2024novel,
  title={A novel deep reinforcement learning framework with BiLSTM-Attention networks for algorithmic trading},
  author={Huang, Y. and others},
  journal={Expert Systems With Applications},
  volume={240},
  year={2024}
}

@article{bhuiyan2025deep,
  title={Deep learning for algorithmic trading: A systematic review of predictive models and optimization strategies},
  author={Bhuiyan, M. S. M. and others},
  journal={Array},
  volume={26},
  year={2025}
}

@article{li2024deepar,
  title={DeepAR-Attention probabilistic prediction for stock price series},
  author={Li, J. and Chen, W. and Zhou, Z. and Yang, J. and Zeng, D.},
  journal={Neural Computing and Applications},
  volume={36},
  year={2024}
}
```
