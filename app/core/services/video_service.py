from app.core.models import VideoTask
from app.utils.storage import minio_client
from app.tasks.video_tasks import process_video_task

class VideoService:
    @staticmethod
    def create_upload_url(user_id: int, filename: str) -> str:
        """生成预签名上传URL"""
        return minio_client.presigned_put_object(
            "uploads",
            f"{user_id}/{filename}",
            expires=3600
        )

    @staticmethod
    def create_video_task(db, user_id: int, video_url: str) -> VideoTask:
        """创建视频处理任务"""
        task = VideoTask(
            user_id=user_id,
            input_path=video_url,
            status=VideoTask.StatusEnum.PENDING
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        
        # 触发异步任务
        process_video_task.delay(task.id)
        return task

    @staticmethod
    def get_task_status(db, task_id: int) -> VideoTask:
        return db.query(VideoTask).filter(VideoTask.id == task_id).first()