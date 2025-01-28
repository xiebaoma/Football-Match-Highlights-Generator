import cv2
import os
from pathlib import Path
from .utils.config import VIDEO_CONFIG

class VideoSlicer:
    def __init__(self):
        self.frame_interval = VIDEO_CONFIG['frame_interval']
        self.clip_duration = VIDEO_CONFIG['clip_duration']
        
    def slice_video(self, video_path: str, output_dir: str):
        """将视频切割为指定时长的片段"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        clip_frames = int(fps * self.clip_duration)
        current_clip = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            if current_clip % clip_frames == 0:
                if current_clip != 0:
                    out.release()
                output_path = os.path.join(
                    output_dir, 
                    f"clip_{int(current_clip/clip_frames)}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    output_path, fourcc, fps, 
                    (int(cap.get(3)), int(cap.get(4)))
                )
            
            out.write(frame)
            current_clip += 1
        
        cap.release()
        return output_dir

    def extract_key_frames(self, video_path: str, output_dir: str):
        """提取关键帧"""
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        count = 0
        
        while success:
            if count % self.frame_interval == 0:
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{count}.jpg"), 
                    image
                )
            success, image = cap.read()
            count += 1
        
        cap.release()
        return output_dir