#!/usr/bin/env python3
"""
Tennis Match Analysis Application
================================

This application analyzes tennis match footage to:
1. Detect tennis court keypoints
2. Calculate bird's eye view homography
3. Track player movements
4. Calculate distance covered by each player

Usage:
    python main.py --input video.mp4 --output results/
"""

import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import our modules
from tennis_analyzer import TennisAnalyzer
from court_detector import CourtDetector
from player_tracker import PlayerTracker

class TennisMatchAnalyzer:
    def __init__(self, model_path: str = 'yolov8x.pt'):
        """
        Initialize the tennis match analyzer with all components
        """
        self.court_detector = CourtDetector()
        self.player_tracker = PlayerTracker(model_path)
        self.tennis_analyzer = TennisAnalyzer(model_path)
        
    def analyze_match(self, video_path: str, output_dir: str = "results") -> Dict:
        """
        Complete analysis of a tennis match video
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all analysis results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/videos", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        print("Starting tennis match analysis...")
        print(f"Input video: {video_path}")
        print(f"Output directory: {output_dir}")
        
        # Step 1: Court Detection and Homography
        print("\n1. Detecting court and calculating homography...")
        court_results = self._detect_court_and_homography(video_path, output_dir)
        
        # Step 2: Player Tracking
        print("\n2. Tracking players...")
        tracking_results = self._track_players(video_path, output_dir)
        
        # Step 3: Distance Analysis
        print("\n3. Analyzing player movements and distances...")
        distance_results = self._analyze_distances(tracking_results, output_dir)
        
        # Step 4: Generate Comprehensive Report
        print("\n4. Generating comprehensive report...")
        final_report = self._generate_comprehensive_report(
            court_results, tracking_results, distance_results, output_dir
        )
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        return final_report
    
    def _detect_court_and_homography(self, video_path: str, output_dir: str) -> Dict:
        """Detect court and calculate homography transformation"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Read first frame for court detection
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Could not read video frame")
        
        # Detect court using advanced detector
        court_corners = self.court_detector.detect_court_keypoints_advanced(frame)
        
        if court_corners is None:
            print("Warning: Could not detect court corners. Using basic detection...")
            court_corners = self.tennis_analyzer.detect_court_keypoints(frame)
        
        if court_corners is None:
            raise ValueError("Could not detect tennis court in video")
        
        # Validate court geometry
        if not self.court_detector.validate_court_geometry(court_corners):
            print("Warning: Detected court geometry may not be accurate")
        
        # Calculate homography matrix
        homography_matrix = self.court_detector.get_court_transform_matrix(court_corners)
        
        # Save court detection visualization
        court_viz = self.court_detector.draw_court_analysis(frame, court_corners)
        cv2.imwrite(f"{output_dir}/court_detection.jpg", court_viz)
        
        # Transform frame to bird's eye view
        if homography_matrix is not None:
            birds_eye_frame = cv2.warpPerspective(frame, homography_matrix, 
                                                (frame.shape[1], frame.shape[0]))
            cv2.imwrite(f"{output_dir}/birds_eye_view.jpg", birds_eye_frame)
        
        return {
            'court_corners': court_corners.tolist(),
            'homography_matrix': homography_matrix.tolist() if homography_matrix is not None else None,
            'court_detection_image': f"{output_dir}/court_detection.jpg",
            'birds_eye_image': f"{output_dir}/birds_eye_view.jpg"
        }
    
    def _track_players(self, video_path: str, output_dir: str) -> Dict:
        """Track players throughout the video"""
        output_video = f"{output_dir}/videos/tracked_players.mp4"
        
        # Process video with player tracking
        tracking_results = self.player_tracker.process_video(video_path, output_video)
        
        # Save tracking data
        tracking_data_file = f"{output_dir}/data/tracking_data.json"
        self.player_tracker.save_tracking_data(tracking_data_file)
        
        return {
            'tracking_results': tracking_results,
            'tracking_video': output_video,
            'tracking_data': tracking_data_file
        }
    
    def _analyze_distances(self, tracking_results: Dict, output_dir: str) -> Dict:
        """Analyze player distances and movements"""
        stats = tracking_results['tracking_results']['track_statistics']
        
        # Calculate additional metrics
        distance_analysis = {}
        for track_id, track_stats in stats.items():
            total_distance = track_stats['total_distance']
            avg_speed = track_stats['average_speed']
            
            # Convert to meters (assuming rough calibration)
            # This would need proper calibration for accurate measurements
            distance_meters = total_distance * 0.01  # Rough conversion
            
            distance_analysis[track_id] = {
                'total_distance_pixels': total_distance,
                'total_distance_meters': distance_meters,
                'average_speed_pixels_per_frame': avg_speed,
                'average_speed_meters_per_second': avg_speed * 0.01 * 30,  # Assuming 30 fps
                'num_detections': track_stats['num_detections']
            }
        
        # Generate movement plots
        self._plot_player_movements(output_dir)
        
        return distance_analysis
    
    def _plot_player_movements(self, output_dir: str):
        """Generate movement trajectory plots"""
        tracks = self.player_tracker.tracks
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Movement trajectories
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (track_id, track) in enumerate(tracks.items()):
            if len(track.positions) > 1:
                positions = np.array(list(track.positions))
                color = colors[i % len(colors)]
                
                ax1.plot(positions[:, 0], positions[:, 1], 
                        color=color, label=f'Player {track_id}', linewidth=2)
                ax1.scatter(positions[0, 0], positions[0, 1], 
                           color=color, s=100, marker='o')
                ax1.scatter(positions[-1, 0], positions[-1, 1], 
                           color=color, s=100, marker='s')
        
        ax1.set_title('Player Movement Trajectories')
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Distance over time
        for i, (track_id, track) in enumerate(tracks.items()):
            if len(track.positions) > 1:
                positions = list(track.positions)
                distances = [0]
                cumulative_distance = 0
                
                for j in range(1, len(positions)):
                    dist = np.linalg.norm(np.array(positions[j]) - np.array(positions[j-1]))
                    cumulative_distance += dist
                    distances.append(cumulative_distance)
                
                color = colors[i % len(colors)]
                ax2.plot(range(len(distances)), distances, 
                        color=color, label=f'Player {track_id}', linewidth=2)
        
        ax2.set_title('Cumulative Distance Over Time')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Distance (pixels)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/player_movements.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comprehensive_report(self, court_results: Dict, tracking_results: Dict, 
                                     distance_results: Dict, output_dir: str) -> Dict:
        """Generate comprehensive analysis report"""
        
        # Compile all results
        report = {
            'analysis_metadata': {
                'timestamp': str(np.datetime64('now')),
                'output_directory': output_dir
            },
            'court_analysis': court_results,
            'player_tracking': tracking_results,
            'distance_analysis': distance_results,
            'summary': {}
        }
        
        # Generate summary statistics
        total_players = len(distance_results)
        total_distance = sum(result['total_distance_meters'] for result in distance_results.values())
        avg_distance = total_distance / total_players if total_players > 0 else 0
        
        report['summary'] = {
            'total_players_detected': total_players,
            'total_distance_covered': total_distance,
            'average_distance_per_player': avg_distance,
            'court_detected': court_results['court_corners'] is not None,
            'homography_calculated': court_results['homography_matrix'] is not None
        }
        
        # Save comprehensive report
        report_file = f"{output_dir}/comprehensive_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate text summary
        self._generate_text_summary(report, output_dir)
        
        return report
    
    def _generate_text_summary(self, report: Dict, output_dir: str):
        """Generate human-readable text summary"""
        summary = report['summary']
        
        text_summary = f"""
TENNIS MATCH ANALYSIS REPORT
============================

Analysis Summary:
- Total players detected: {summary['total_players_detected']}
- Total distance covered: {summary['total_distance_covered']:.2f} meters
- Average distance per player: {summary['average_distance_per_player']:.2f} meters
- Court detection: {'Successful' if summary['court_detected'] else 'Failed'}
- Homography calculation: {'Successful' if summary['homography_calculated'] else 'Failed'}

Player Details:
"""
        
        for track_id, stats in report['distance_analysis'].items():
            text_summary += f"""
Player {track_id}:
- Total distance: {stats['total_distance_meters']:.2f} meters
- Average speed: {stats['average_speed_meters_per_second']:.2f} m/s
- Number of detections: {stats['num_detections']}
"""
        
        text_summary += f"""

Output Files:
- Court detection: {report['court_analysis']['court_detection_image']}
- Bird's eye view: {report['court_analysis']['birds_eye_image']}
- Player tracking video: {report['player_tracking']['tracking_video']}
- Movement plots: {output_dir}/plots/player_movements.png
- Tracking data: {report['player_tracking']['tracking_data']}
- Comprehensive report: {output_dir}/comprehensive_analysis_report.json
"""
        
        # Save text summary
        with open(f"{output_dir}/analysis_summary.txt", 'w') as f:
            f.write(text_summary)
        
        # Print summary to console
        print(text_summary)

def main():
    """Main function to run the tennis match analyzer"""
    parser = argparse.ArgumentParser(description='Analyze tennis match footage')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input video file path')
    parser.add_argument('--output', '-o', default='results',
                       help='Output directory for results')
    parser.add_argument('--model', '-m', default='yolov8x.pt',
                       help='YOLO model path')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    try:
        # Initialize analyzer
        analyzer = TennisMatchAnalyzer(args.model)
        
        # Run analysis
        results = analyzer.analyze_match(args.input, args.output)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 