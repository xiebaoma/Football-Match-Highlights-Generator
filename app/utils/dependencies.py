from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from app.core.models.user_model import User
from app.utils.config import settings
from app.utils.security import verify_token
from app.utils.database import SessionLocal

# OAuth2 认证方案
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# 数据库会话依赖项
def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话的依赖项。
    使用 `yield` 确保会话在使用后正确关闭。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 获取当前用户依赖项
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    从 JWT 令牌中获取当前用户。
    如果令牌无效或用户不存在，抛出 401 未授权异常。
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_token(token)
        if payload is None:
            raise credentials_exception
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user