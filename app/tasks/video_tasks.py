from .celery_app import celery
from app.core.models import VideoTask
from app.utils.dependencies import get_db
from app.utils.video import download_video, analyze_video, generate_highlight_video

@celery.task(bind=True)
def process_video_task(self, task_id: int):
    db = next(get_db())
    
    try:
        task = db.query(VideoTask).get(task_id)
        task.status = VideoTask.StatusEnum.PROCESSING
        db.commit()
        
        # 实际处理逻辑
        video_path = download_video(task.input_path)
        highlights = analyze_video(video_path)
        output_url = generate_highlight_video(highlights)
        
        task.status = VideoTask.StatusEnum.COMPLETED
        task.output_path = output_url
    except Exception as e:
        task.status = VideoTask.StatusEnum.FAILED
        task.error_message = str(e)
    finally:
        db.commit()
        db.close()