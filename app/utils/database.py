from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.utils.config import settings

# 数据库连接 URL
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # 自动检测连接是否有效
    pool_size=10,        # 连接池大小
    max_overflow=20      # 连接池溢出大小
)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()