"""
Algo Trading System - FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import trading, backtest, profile
from app.config import settings

app = FastAPI(
    title="Algo Trading System",
    description="AI-Powered Algorithmic Trading Platform for Indian Markets (NSE/BSE)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trading.router, prefix="/api/v1/trading", tags=["Trading"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtest"])
app.include_router(profile.router, prefix="/api/v1/profile", tags=["Profile"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Algo Trading System API",
        "version": "1.0.0",
        "target_market": "India (NSE/BSE)",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
        }
    }
