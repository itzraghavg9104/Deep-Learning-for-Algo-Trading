"""
Application configuration settings.
"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/algotrading"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Market Data
    DEFAULT_MARKET: str = "NSE"  # NSE or BSE
    
    # News API (Optional - for sentiment)
    NEWS_API_KEY: str = ""
    USE_SENTIMENT: bool = False
    
    # ML Models
    MODEL_PATH: str = "./models"
    DEEPAR_MODEL: str = "deepar_v1.pt"
    PPO_MODEL: str = "ppo_agent_v1.zip"
    
    # JWT
    JWT_SECRET: str = "jwt-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
