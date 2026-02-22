from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

Base = declarative_base()

# In demo mode we avoid initializing a Postgres engine entirely.
if settings.DEMO_MODE:
    SessionLocal = None

    def get_engine():
        return None

    # Dependency to get DB session (None in demo mode)
    def get_db():
        yield None
else:
    # Use DATABASE_URL from settings. Normalize async URL for sync engine consumers.
    SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL.replace(
        "postgresql+asyncpg://",
        "postgresql://",
    )

    def get_engine():
        return create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

    # Dependency to get DB session
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
