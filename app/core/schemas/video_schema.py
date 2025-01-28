from pydantic import BaseModel
from datetime import datetime

class TaskCreate(BaseModel):
    video_url: str

class TaskResponse(BaseModel):
    task_id: int
    status: str
    created_at: datetime
    result_url: str | None
    
    class Config:
        orm_mode = True