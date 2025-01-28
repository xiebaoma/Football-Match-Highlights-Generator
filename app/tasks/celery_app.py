from celery import Celery
from app.utils.config import settings

celery = Celery(
    "video_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery.conf.update(
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle"]
)