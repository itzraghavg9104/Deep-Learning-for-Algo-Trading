# 4. ECONOMIC ANALYSIS

## 4.1. Cost Structure

The project is economically efficient due to open-source-first design and minimal infrastructure requirements:

| Cost Category | Estimated Cost | Details |
|--------------|---------------|---------|
| Software Licensing | $0 (INR 0) | Entire stack (Python, PyTorch, SB3, FastAPI, Next.js) is open-source |
| Data Access | $0 (INR 0) | yfinance provides free NSE/BSE data via Yahoo Finance API |
| Development Hardware | ~$600 (existing laptop) | Standard laptop with 8GB+ RAM; no GPU required |
| Cloud Infrastructure | $0 (demo), ~$6/mo (production) | Optional: VPS for PostgreSQL + backend hosting |
| Firebase Services | $0 (Spark plan) | Free tier: 50K reads/day, 20K writes/day, 10K auth users |
| Docker Deployment | $0 | Docker Compose runs locally; no paid images |

## 4.2. Effort and Resource Allocation

The project was developed by a team of 4 students over approximately 16 weeks:

| Phase | Duration | Activities |
|-------|----------|------------|
| Research and Design | 3 weeks | Literature survey, architecture design, tech stack selection |
| Backend Development | 5 weeks | FastAPI routes, Layer 1/2/3 implementation, ML pipeline, services |
| Frontend Development | 4 weeks | Next.js pages, auth flow, dashboard, WebSocket integration, charts |
| Integration and Testing | 2 weeks | API-frontend integration, Docker setup, mode testing |
| Documentation | 2 weeks | Technical docs, report writing, diagrams |

Total estimated person-hours: ~800 hours (200 per team member).

## 4.3. Economic Feasibility Outcome

Given negligible software procurement costs, free access to training data, and configurable runtime requirements (demo mode requires zero infrastructure), the platform is economically feasible for academic deployment, classroom demonstrations, and pilot experimentation. The primary cost is human effort, which is typical for academic research projects. For production deployment, the main additional costs would be cloud hosting ($6-25/month) and optional live market data subscriptions ($25-125/month for real-time feeds).
