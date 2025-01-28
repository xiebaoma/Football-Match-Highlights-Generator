from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:pass@localhost/db"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    JWT_SECRET_KEY: str
    JWT_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()