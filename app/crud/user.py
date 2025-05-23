from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
from contextlib import asynccontextmanager
import uuid

from app.core.security import hash_password, verify_password
from app.db.sql import sql

@asynccontextmanager
async def get_session():
    async with AsyncSession(sql) as session:
        yield session

async def get_user_by_email(session: AsyncSession, email: str):
    print(f"Fetching user by email: {email}")
    query = text("""
        SELECT u.*
        FROM users u
        JOIN user_identifiers ui ON ui.userid = u.userid
        WHERE ui.value = :email AND ui.type = 'email'
        LIMIT 1
    """)
    result = await session.execute(query, {"email": email})
    row = result.fetchone()
    return dict(row._mapping) if row else None

async def register_user(session: AsyncSession, username: str, email: str, password: str):
    # 1. Check if email is already registered
    if await get_user_by_email(session, email):
        raise ValueError("Email already registered.")

    # 2. Check if username already exists
    username_check = await session.execute(
        text("SELECT 1 FROM users WHERE username = :username LIMIT 1"),
        {"username": username}
    )
    if username_check.fetchone():
        raise ValueError("Username already taken.")

    # 3. Generate UUIDs
    user_id = str(uuid.uuid4())
    identifier_id = str(uuid.uuid4())

    # 4. Insert into users table
    await session.execute(
        text("""
            INSERT INTO users (userid, username, password)
            VALUES (:userid, :username, :password)
        """),
        {
            "userid": user_id,
            "username": username,
            "password": hash_password(password)
        }
    )

    # 5. Insert into user_identifiers table
    await session.execute(
        text("""
            INSERT INTO user_identifiers (id, userid, type, value)
            VALUES (:id, :userid, 'email', :email)
        """),
        {
            "id": identifier_id,
            "userid": user_id,
            "email": email
        }
    )

    # 6. Commit the transaction
    await session.commit()

    # 7. Optionally return the new user
    return {
        "userid": user_id,
        "username": username,
        "email": email
    }

async def login_user(session: AsyncSession, email: str, password: str):
    query = text("""
        SELECT u.*
        FROM users u
        JOIN user_identifiers ui ON ui.userid = u.userid
        WHERE ui.value = :email AND ui.type = 'email'
        LIMIT 1
    """)
    result = await session.execute(query, {"email": email})
    row = result.fetchone()

    if not row:
        raise ValueError("Invalid credentials.")

    user_data = dict(row._mapping)

    # If your `User` model supports instantiation like this
    user = User(**user_data)

    if not verify_password(password, user.password):
        raise ValueError("Invalid credentials.")

    return user
