import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
import pandas as pd
from tqdm import tqdm
import os
from typing import List, Tuple, Dict, Optional
import json

class TennisAnalyzer:
    def __init__(self, model_path: str = 'yolov8x.pt'):
        """
        Initialize the tennis analyzer with YOLO model for player detection
        """
        self.model = YOLO(model_path)
        self.court_keypoints = None
        self.homography_matrix = None
        self.player_tracks = {}  # Dictionary to store player tracking data
        self.frame_count = 0
        
    def detect_court_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect tennis court keypoints using color and line detection
        Returns: 4 keypoints representing court corners
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for court lines (white/light green)
        lower_court = np.array([35, 20, 20])  # Green court
        upper_court = np.array([85, 255, 255])
        
        # Create mask for court area
        court_mask = cv2.inRange(hsv, lower_court, upper_court)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
            
        # Find the largest contour (should be the court)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to get corners
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have 4 points, use them as keypoints
        if len(approx) == 4:
            keypoints = approx.reshape(-1, 2)
            return keypoints
        
        # If not exactly 4 points, find the bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        return box
    
    def calculate_homography(self, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """
        Calculate homography matrix for bird's eye view transformation
        """
        if len(source_points) != 4 or len(target_points) != 4:
            return None
            
        # Calculate homography matrix
        H = cv2.findHomography(source_points, target_points)[0]
        return H
    
    def transform_to_birds_eye(self, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Transform frame to bird's eye view using homography matrix
        """
        if H is None:
            return frame
            
        height, width = frame.shape[:2]
        transformed = cv2.warpPerspective(frame, H, (width, height))
        return transformed
    
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in the frame using YOLO
        """
        results = self.model(frame, verbose=False)
        players = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for person class (class 0 in COCO dataset)
                    if box.cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        if conf > 0.5:  # Confidence threshold
                            players.append({
                                'bbox': [x1, y1, x2, y2],
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                'confidence': conf
                            })
        
        return players
    
    def track_players(self, players: List[Dict], frame_number: int) -> Dict:
        """
        Track players across frames using simple centroid tracking
        """
        if frame_number == 0:
            # Initialize tracking for first frame
            for i, player in enumerate(players):
                self.player_tracks[f'player_{i}'] = {
                    'positions': [player['center']],
                    'distances': [0],
                    'frame_numbers': [frame_number]
                }
            return self.player_tracks
        
        # Simple tracking: assign players to closest existing tracks
        current_tracks = {}
        
        for player in players:
            min_distance = float('inf')
            best_track = None
            
            for track_id, track_data in self.player_tracks.items():
                if len(track_data['positions']) > 0:
                    last_pos = track_data['positions'][-1]
                    distance = euclidean(player['center'], last_pos)
                    
                    if distance < min_distance and distance < 100:  # Max distance threshold
                        min_distance = distance
                        best_track = track_id
            
            if best_track:
                # Update existing track
                last_pos = self.player_tracks[best_track]['positions'][-1]
                distance = euclidean(player['center'], last_pos)
                
                self.player_tracks[best_track]['positions'].append(player['center'])
                self.player_tracks[best_track]['distances'].append(distance)
                self.player_tracks[best_track]['frame_numbers'].append(frame_number)
                
                current_tracks[best_track] = player
            else:
                # Create new track
                track_id = f'player_{len(self.player_tracks)}'
                self.player_tracks[track_id] = {
                    'positions': [player['center']],
                    'distances': [0],
                    'frame_numbers': [frame_number]
                }
                current_tracks[track_id] = player
        
        return current_tracks
    
    def calculate_total_distance(self, track_id: str) -> float:
        """
        Calculate total distance covered by a player
        """
        if track_id not in self.player_tracks:
            return 0.0
            
        track = self.player_tracks[track_id]
        return sum(track['distances'])
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict:
        """
        Process tennis match video and return analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize output video writer if output path is provided
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        court_detected = False
        
        print("Processing video...")
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect court keypoints (only once for efficiency)
                if not court_detected and self.court_keypoints is None:
                    self.court_keypoints = self.detect_court_keypoints(frame)
                    if self.court_keypoints is not None:
                        court_detected = True
                        # Define target points for bird's eye view
                        height, width = frame.shape[:2]
                        target_points = np.array([
                            [0, 0],
                            [width, 0],
                            [width, height],
                            [0, height]
                        ], dtype=np.float32)
                        
                        self.homography_matrix = self.calculate_homography(
                            self.court_keypoints.astype(np.float32), 
                            target_points
                        )
                
                # Detect players
                players = self.detect_players(frame)
                
                # Track players
                current_tracks = self.track_players(players, frame_count)
                
                # Draw court keypoints and player bounding boxes
                if self.court_keypoints is not None:
                    # Draw court keypoints
                    for point in self.court_keypoints:
                        cv2.circle(frame, tuple(point), 10, (0, 255, 0), -1)
                    
                    # Draw court outline
                    cv2.polylines(frame, [self.court_keypoints], True, (0, 255, 0), 2)
                
                # Draw player bounding boxes and tracking info
                for track_id, player in current_tracks.items():
                    x1, y1, x2, y2 = player['bbox']
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    # Add player ID and distance
                    total_dist = self.calculate_total_distance(track_id)
                    cv2.putText(frame, f"{track_id}: {total_dist:.1f}m", 
                              (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 0, 0), 2)
                
                # Write frame to output video
                if out_writer:
                    out_writer.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        if out_writer:
            out_writer.release()
        
        # Calculate final statistics
        analysis_results = self.generate_analysis_report()
        
        return analysis_results
    
    def generate_analysis_report(self) -> Dict:
        """
        Generate comprehensive analysis report
        """
        report = {
            'total_frames': self.frame_count,
            'players': {}
        }
        
        for track_id, track_data in self.player_tracks.items():
            total_distance = sum(track_data['distances'])
            avg_speed = total_distance / len(track_data['distances']) if len(track_data['distances']) > 0 else 0
            
            report['players'][track_id] = {
                'total_distance': total_distance,
                'average_speed': avg_speed,
                'positions': track_data['positions'],
                'frame_numbers': track_data['frame_numbers']
            }
        
        return report
    
    def save_analysis_results(self, results: Dict, output_file: str):
        """
        Save analysis results to JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def plot_player_movements(self, output_file: str = None):
        """
        Plot player movement trajectories
        """
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (track_id, track_data) in enumerate(self.player_tracks.items()):
            if len(track_data['positions']) > 1:
                positions = np.array(track_data['positions'])
                color = colors[i % len(colors)]
                
                plt.plot(positions[:, 0], positions[:, 1], 
                        color=color, label=track_id, linewidth=2)
                plt.scatter(positions[0, 0], positions[0, 1], 
                           color=color, s=100, marker='o', label=f'{track_id} Start')
                plt.scatter(positions[-1, 0], positions[-1, 1], 
                           color=color, s=100, marker='s', label=f'{track_id} End')
        
        plt.title('Player Movement Trajectories')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.legend()
        plt.grid(True)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

# Main function removed - use main.py for application entry point 