from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings


ASYNC_SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL.replace(
    "postgresql://",
    "postgresql+asyncpg://",
)

engine = create_async_engine(
    ASYNC_SQLALCHEMY_DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
