from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.utils.config import Base

class VideoTask(Base):
    __tablename__ = "video_tasks"
    
    class StatusEnum:
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(StatusEnum), default=StatusEnum.PENDING)
    input_path = Column(String(1024))
    output_path = Column(String(1024))
    error_message = Column(String(2048))
    
    user = relationship("User", back_populates="tasks")