from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.user import RegisterRequest, LoginRequest, TokenResponse
from app.crud.user import register_user, login_user

router = APIRouter()

@router.post("/register", response_model=TokenResponse)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await register_user(db, data.username, data.email, data.password)
        return TokenResponse(access_token=str(user.userid))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await login_user(db, data.email, data.password)
        return TokenResponse(access_token=str(user.userid))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
