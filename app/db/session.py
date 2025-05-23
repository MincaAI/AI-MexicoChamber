from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.db.sql import sql

# Session maker bound to the `sql` engine
async_session = sessionmaker(
    bind=sql, class_=AsyncSession, expire_on_commit=False
)

# Dependency for FastAPI routes
async def get_db():
    async with async_session() as session:
        yield session
