from fastapi import APIRouter, Depends, HTTPException
from app.core.schemas.user_schema import UserCreate, UserResponse
from app.core.services.auth_service import AuthService
from app.utils.dependencies import get_db
from app.utils.database import get_db
from app.core.models.user_model import User
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db=Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(400, "Email already registered")
    
    new_user = User(
        email=user.email,
        hashed_password=AuthService.get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    return new_user

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db: Session = Depends(get_db)  # 注入 db
    user = AuthService.authenticate_user(
        db, form_data.username, form_data.password
    )
    if not user:
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": AuthService.create_user_token(user)}