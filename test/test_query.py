from sqlalchemy import text
from app.db.sql import sql

async def test_query():
    async with sql.connect() as conn:
        result = await conn.execute(text("SELECT 'hello from db'"))
        return result.scalar()  # or .fetchall()