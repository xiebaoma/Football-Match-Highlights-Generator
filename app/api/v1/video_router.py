from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.schemas.video_schema import TaskCreate, TaskResponse
from app.core.services import VideoService
from app.utils.dependencies import get_current_user, get_db

router = APIRouter()

@router.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    data: TaskCreate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)  # 注入 db
):
    task = VideoService.create_video_task(db, user.id, data.video_url)
    return {"task_id": task.id}

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)  # 注入 db
):
    task = VideoService.get_task_status(db, task_id)
    if task.user_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")
    return task