# app/db/sql.py

import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine,async_sessionmaker

load_dotenv()

tmpPostgres = urlparse(os.getenv("DATABASE_URL"))

DATABASE_URL = (
    f"postgresql+asyncpg://{tmpPostgres.username}:{tmpPostgres.password}@"
    f"{tmpPostgres.hostname}{tmpPostgres.path}?ssl=require"
)

# Exported variable: `sql`
sql: AsyncEngine = create_async_engine(DATABASE_URL, echo=True)
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)