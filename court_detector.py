import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

class CourtDetector:
    def __init__(self):
        """
        Initialize the tennis court detector
        """
        self.court_dimensions = {
            'singles_width': 8.23,  # meters
            'singles_length': 23.77,  # meters
            'doubles_width': 10.97,  # meters
            'doubles_length': 23.77   # meters
        }
        
    def detect_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect tennis court lines using edge detection and Hough line transform
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply morphological operations to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        return lines
    
    def filter_court_lines(self, lines: np.ndarray, frame_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Filter lines to identify potential court boundaries
        """
        if lines is None:
            return []
        
        height, width = frame_shape[:2]
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)
            
            # Classify lines as horizontal or vertical
            if angle < 30 or angle > 150:  # Horizontal lines
                horizontal_lines.append(line[0])
            elif 60 < angle < 120:  # Vertical lines
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def find_court_corners(self, horizontal_lines: List, vertical_lines: List, 
                          frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Find court corners by intersecting horizontal and vertical lines
        """
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        height, width = frame_shape[:2]
        intersections = []
        
        # Find intersections between horizontal and vertical lines
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = self.line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    # Check if intersection is within frame bounds
                    if 0 <= x <= width and 0 <= y <= height:
                        intersections.append(intersection)
        
        if len(intersections) < 4:
            return None
        
        # Find the 4 corners that form the largest rectangle
        corners = self.find_best_rectangle(intersections)
        
        return np.array(corners, dtype=np.int32) if corners else None
    
    def line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Calculate intersection point of two lines
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denominator) < 1e-10:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def find_best_rectangle(self, points: List[Tuple[float, float]]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the best rectangle from intersection points
        """
        if len(points) < 4:
            return None
        
        # Convert to numpy array for easier processing
        points = np.array(points)
        
        # Find the convex hull
        hull = cv2.convexHull(points)
        
        # Approximate the hull to get corners
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # If we have exactly 4 points, return them
        if len(approx) == 4:
            return [tuple(point[0]) for point in approx]
        
        # If we have more than 4 points, try to find the best 4
        if len(approx) > 4:
            # Find the 4 points that form the largest area
            best_area = 0
            best_corners = None
            
            # Try different combinations of 4 points
            for i in range(len(approx) - 3):
                for j in range(i + 1, len(approx) - 2):
                    for k in range(j + 1, len(approx) - 1):
                        for l in range(k + 1, len(approx)):
                            corners = [approx[i][0], approx[j][0], approx[k][0], approx[l][0]]
                            area = cv2.contourArea(np.array(corners))
                            
                            if area > best_area:
                                best_area = area
                                best_corners = corners
            
            if best_corners:
                return [tuple(corner) for corner in best_corners]
        
        return None
    
    def detect_court_keypoints_advanced(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Advanced court detection using line detection and geometry
        """
        # Detect lines
        lines = self.detect_court_lines(frame)
        
        if lines is None:
            return None
        
        # Filter lines
        horizontal_lines, vertical_lines = self.filter_court_lines(lines, frame.shape)
        
        # Find court corners
        corners = self.find_court_corners(horizontal_lines, vertical_lines, frame.shape)
        
        return corners
    
    def validate_court_geometry(self, corners: np.ndarray) -> bool:
        """
        Validate that the detected corners form a reasonable tennis court shape
        """
        if corners is None or len(corners) != 4:
            return False
        
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.linalg.norm(np.array(p2) - np.array(p1))
            sides.append(length)
        
        # Check aspect ratio (tennis courts are roughly 2.9:1 for singles)
        width = min(sides[0], sides[2])
        length = max(sides[0], sides[2])
        
        aspect_ratio = length / width if width > 0 else 0
        
        # Tennis court aspect ratio should be around 2.9
        return 2.0 < aspect_ratio < 4.0
    
    def draw_court_analysis(self, frame: np.ndarray, corners: np.ndarray, 
                           lines: np.ndarray = None) -> np.ndarray:
        """
        Draw court analysis visualization
        """
        result = frame.copy()
        
        # Draw detected corners
        if corners is not None:
            for i, corner in enumerate(corners):
                cv2.circle(result, tuple(corner), 10, (0, 255, 0), -1)
                cv2.putText(result, f"C{i}", tuple(corner), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw court outline
            cv2.polylines(result, [corners], True, (0, 255, 0), 3)
        
        # Draw detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return result
    
    def get_court_transform_matrix(self, corners: np.ndarray, 
                                 target_size: Tuple[int, int] = (800, 600)) -> Optional[np.ndarray]:
        """
        Calculate transformation matrix for bird's eye view
        """
        if corners is None or len(corners) != 4:
            return None
        
        # Define target points for bird's eye view
        target_width, target_height = target_size
        target_points = np.array([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ], dtype=np.float32)
        
        # Calculate homography matrix
        H = cv2.findHomography(corners.astype(np.float32), target_points)
        
        return H[0] if H is not None else None

# Main function removed - use main.py for application entry point 