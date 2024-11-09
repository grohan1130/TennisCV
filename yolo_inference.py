from ultralytics import YOLO

model = YOLO('yolov8x')

model.predict('input_videos/ao_input_video_1.mp4', save = True)