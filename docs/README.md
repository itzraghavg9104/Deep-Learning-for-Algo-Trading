# Deep Learning for Algo Trading — Technical Documentation

Comprehensive, codebase-aligned documentation for the AI-powered algorithmic trading platform targeting Indian equity markets (NSE/BSE).

## Document Index

| # | Document | Description |
|---|----------|-------------|
| 1 | [System Overview](./01_system_overview.md) | Project purpose, architecture, tech stack, runtime modes, repo structure |
| 2 | [Backend Architecture](./02_backend_architecture.md) | Application lifecycle, config, all API routes, Layer 1/2/3, services, storage |
| 3 | [Frontend Architecture](./03_frontend_architecture.md) | Next.js pages, auth flow, state management, component hierarchy, WebSocket |
| 4 | [ML Pipeline and Models](./04_ml_pipeline_and_models.md) | Data download, LSTM training, PPO training, DeepAR, inference, per-user retraining |
| 5 | [API and WebSocket Contract](./05_api_and_websocket_contract.md) | Complete endpoint reference with request/response schemas, error codes |
| 6 | [Deployment, Configuration, and Limits](./06_deployment_configuration_and_limits.md) | Setup guide, Docker, env vars, known limitations, troubleshooting |

## Quick Reference

- **Backend**: Python 3.12, FastAPI, PyTorch (LSTM), Stable-Baselines3 (PPO), Gymnasium
- **Frontend**: Next.js 16, React 19, TailwindCSS 4, Zustand, Recharts
- **Infra**: PostgreSQL 15, Redis 7, Docker Compose
- **Models**: LSTM (price prediction), PPO (trading decisions), DeepAR (experimental)
- **API Base**: `/api/v1`, docs at `/docs`
- **Auth**: Firebase SDK (frontend) / JWT + Firebase Admin (backend)

## Diagram References

Architecture and flow diagrams are available in the [`images/`](./images/) subdirectory:
- `system_architecture.png`
- `data_flow.png`
- `training_pipeline.png`
- `lstm_architecture.png`
- `ppo_training.png`
- `risk_profiler.png`
