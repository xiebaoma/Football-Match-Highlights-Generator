from datetime import datetime, timedelta
from jose import JWTError, jwt
from app.utils.config import settings

def create_access_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(
        minutes=settings.JWT_EXPIRE_MINUTES
    )
    return jwt.encode(
        {**data, "exp": expire},
        settings.JWT_SECRET_KEY,
        algorithm="HS256"
    )

def verify_token(token: str) -> dict:
    try:
        return jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=["HS256"]
        )
    except JWTError:
        return None