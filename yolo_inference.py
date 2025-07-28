from ultralytics import YOLO
import cv2
import numpy as np
import os

def run_yolo_inference(video_path: str, output_path: str = None, model_path: str = 'yolov8x.pt'):
    """
    Run YOLO inference on tennis match video
    
    Args:
        video_path: Path to input video
        output_path: Path for output video (optional)
        model_path: Path to YOLO model
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    if output_path:
        # Save results to video
        results = model.predict(video_path, save=True, project='output', name='yolo_inference')
        print(f"Results saved to: {output_path}")
    else:
        # Just run inference without saving
        results = model.predict(video_path, save=False)
        print(f"Processed {len(results)} frames")
    
    return results

def analyze_yolo_results(results):
    """
    Analyze YOLO detection results
    
    Args:
        results: YOLO results object
    """
    total_detections = 0
    person_detections = 0
    
    for result in results:
        if result.boxes is not None:
            total_detections += len(result.boxes)
            
            # Count person detections (class 0)
            for box in result.boxes:
                if box.cls == 0:  # Person class
                    person_detections += 1
    
    print(f"Total detections: {total_detections}")
    print(f"Person detections: {person_detections}")
    
    return {
        'total_detections': total_detections,
        'person_detections': person_detections
    }

# Example usage removed - use main.py for application entry point