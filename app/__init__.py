from app.utils.database import Base, engine

def init_db():
    """初始化数据库，创建所有表"""
    Base.metadata.create_all(bind=engine)

# 在应用启动时调用
init_db()