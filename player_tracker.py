import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class PlayerDetection:
    """Data class for player detection results"""
    bbox: List[float]  # [x1, y1, x2, y2]
    center: Tuple[float, float]
    confidence: float
    class_id: int = 0  # Person class

@dataclass
class PlayerTrack:
    """Data class for player tracking information"""
    track_id: int
    detections: deque
    positions: deque
    velocities: deque
    last_update: float
    total_distance: float
    is_active: bool
    color: Tuple[int, int, int]
    
    def __post_init__(self):
        self.detections = deque(maxlen=30)  # Keep last 30 detections
        self.positions = deque(maxlen=30)
        self.velocities = deque(maxlen=30)

class PlayerTracker:
    def __init__(self, model_path: str = 'yolov8x.pt', max_disappeared: int = 30):
        """
        Initialize the player tracker
        
        Args:
            model_path: Path to YOLO model
            max_disappeared: Maximum frames a player can be missing before being removed
        """
        self.model = YOLO(model_path)
        self.tracks = {}
        self.next_track_id = 0
        self.max_disappeared = max_disappeared
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
    def detect_players(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[PlayerDetection]:
        """
        Detect players in the frame using YOLO
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for person class (class 0 in COCO dataset)
                    if box.cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        if conf > confidence_threshold:
                            detection = PlayerDetection(
                                bbox=[x1, y1, x2, y2],
                                center=((x1 + x2) / 2, (y1 + y2) / 2),
                                confidence=conf
                            )
                            detections.append(detection)
        
        return detections
    
    def update_tracks(self, detections: List[PlayerDetection], frame_number: int) -> Dict[int, PlayerTrack]:
        """
        Update player tracks with new detections
        """
        current_time = time.time()
        
        # If no tracks exist, initialize new tracks
        if len(self.tracks) == 0:
            for detection in detections:
                self._create_new_track(detection, current_time)
            return self.tracks
        
        # If no detections, mark all tracks as inactive
        if len(detections) == 0:
            for track_id in self.tracks:
                self.tracks[track_id].is_active = False
            return self.tracks
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        # Calculate distances between all tracks and detections
        for track_id, track in self.tracks.items():
            if not track.is_active:
                continue
                
            if len(track.positions) == 0:
                continue
                
            last_pos = track.positions[-1]
            
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                distance = euclidean(last_pos, detection.center)
                
                # If distance is small enough, consider it a match
                if distance < 100:  # Max distance threshold
                    # Update track
                    self._update_track(track_id, detection, current_time)
                    matched_tracks.add(track_id)
                    matched_detections.add(i)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self._create_new_track(detection, current_time)
        
        # Mark unmatched tracks as inactive
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks and track.is_active:
                track.is_active = False
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return self.tracks
    
    def _create_new_track(self, detection: PlayerDetection, current_time: float):
        """Create a new track for a detection"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        color = self.colors[track_id % len(self.colors)]
        
        track = PlayerTrack(
            track_id=track_id,
            detections=deque(maxlen=30),
            positions=deque(maxlen=30),
            velocities=deque(maxlen=30),
            last_update=current_time,
            total_distance=0.0,
            is_active=True,
            color=color
        )
        
        track.detections.append(detection)
        track.positions.append(detection.center)
        track.velocities.append((0.0, 0.0))  # Initial velocity
        
        self.tracks[track_id] = track
    
    def _update_track(self, track_id: int, detection: PlayerDetection, current_time: float):
        """Update an existing track with a new detection"""
        track = self.tracks[track_id]
        
        # Calculate velocity
        if len(track.positions) > 0:
            last_pos = track.positions[-1]
            velocity = (
                detection.center[0] - last_pos[0],
                detection.center[1] - last_pos[1]
            )
        else:
            velocity = (0.0, 0.0)
        
        # Calculate distance
        if len(track.positions) > 0:
            distance = euclidean(detection.center, track.positions[-1])
            track.total_distance += distance
        
        # Update track
        track.detections.append(detection)
        track.positions.append(detection.center)
        track.velocities.append(velocity)
        track.last_update = current_time
        track.is_active = True
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated recently"""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if not track.is_active and (current_time - track.last_update) > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_active_tracks(self) -> Dict[int, PlayerTrack]:
        """Get only active tracks"""
        return {track_id: track for track_id, track in self.tracks.items() if track.is_active}
    
    def get_track_statistics(self) -> Dict[int, Dict]:
        """Get statistics for all tracks"""
        stats = {}
        
        for track_id, track in self.tracks.items():
            if len(track.positions) < 2:
                continue
                
            # Calculate average velocity
            velocities = list(track.velocities)
            avg_velocity = np.mean(velocities, axis=0) if velocities else (0.0, 0.0)
            
            # Calculate total distance
            total_distance = track.total_distance
            
            # Calculate average speed
            avg_speed = np.linalg.norm(avg_velocity) if len(velocities) > 0 else 0.0
            
            stats[track_id] = {
                'total_distance': total_distance,
                'average_speed': avg_speed,
                'average_velocity': avg_velocity.tolist(),
                'num_detections': len(track.detections),
                'is_active': track.is_active,
                'color': track.color
            }
        
        return stats
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracks and player bounding boxes on frame"""
        result = frame.copy()
        
        for track_id, track in self.tracks.items():
            if not track.is_active or len(track.positions) == 0:
                continue
            
            # Draw track trail
            if len(track.positions) > 1:
                points = np.array(list(track.positions), dtype=np.int32)
                cv2.polylines(result, [points], False, track.color, 2)
            
            # Draw current position
            current_pos = track.positions[-1]
            cv2.circle(result, (int(current_pos[0]), int(current_pos[1])), 
                      5, track.color, -1)
            
            # Draw bounding box and info
            if len(track.detections) > 0:
                latest_detection = track.detections[-1]
                x1, y1, x2, y2 = latest_detection.bbox
                
                # Draw bounding box
                cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), 
                             track.color, 2)
                
                # Draw track info
                info_text = f"ID: {track_id} | Dist: {track.total_distance:.1f}"
                cv2.putText(result, info_text, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2)
        
        return result
    
    def save_tracking_data(self, output_file: str):
        """Save tracking data to JSON file"""
        data = {
            'tracks': {}
        }
        
        for track_id, track in self.tracks.items():
            data['tracks'][track_id] = {
                'track_id': track.track_id,
                'positions': list(track.positions),
                'velocities': list(track.velocities),
                'total_distance': track.total_distance,
                'is_active': track.is_active,
                'color': track.color
            }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict:
        """
        Process video and track players
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize output video writer
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print("Processing video for player tracking...")
        from tqdm import tqdm
        
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect players
                detections = self.detect_players(frame)
                
                # Update tracks
                self.update_tracks(detections, frame_count)
                
                # Draw tracks
                result_frame = self.draw_tracks(frame)
                
                # Write frame
                if out_writer:
                    out_writer.write(result_frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        if out_writer:
            out_writer.release()
        
        # Get final statistics
        stats = self.get_track_statistics()
        
        return {
            'total_frames': frame_count,
            'track_statistics': stats
        }

# Main function removed - use main.py for application entry point 