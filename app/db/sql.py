# app/db/sql.py

import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

load_dotenv()

tmpPostgres = urlparse(os.getenv("DATABASE_URL"))

DATABASE_URL = (
    f"postgresql+asyncpg://{tmpPostgres.username}:{tmpPostgres.password}@"
    f"{tmpPostgres.hostname}{tmpPostgres.path}?ssl=require"
)

# Exported variable: `sql`
sql: AsyncEngine = create_async_engine(DATABASE_URL, echo=True)
