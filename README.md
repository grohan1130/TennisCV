# Tennis Match Analysis Application

A comprehensive computer vision application for analyzing tennis match footage. This application detects tennis court keypoints, calculates bird's eye view homography, tracks player movements, and calculates distance covered by each player.

## Features

### 🏟️ Court Detection

- **Advanced Line Detection**: Uses Hough line transform to detect court boundaries
- **Geometric Validation**: Validates detected court geometry against standard tennis court dimensions
- **Corner Detection**: Identifies court corners for homography calculation

### 🎯 Player Tracking

- **YOLO-based Detection**: Uses YOLOv8 for accurate player detection
- **Multi-Object Tracking**: Tracks multiple players simultaneously
- **Trajectory Analysis**: Records player movement paths and velocities
- **Distance Calculation**: Computes total distance covered by each player

### 📊 Analysis & Visualization

- **Bird's Eye View**: Transforms court perspective for better movement analysis
- **Movement Plots**: Generates trajectory and distance-over-time visualizations
- **Comprehensive Reports**: Provides detailed analysis in JSON and text formats
- **Video Output**: Creates annotated videos showing tracking results

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Setup

1. **Clone the repository**:

```bash
git clone <repository-url>
cd TennisCV
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download YOLO model** (if not already present):

```bash
# The application will automatically download YOLOv8x if not found
# Or manually download from: https://github.com/ultralytics/ultralytics
```

## Usage

### Basic Usage

```bash
python main.py --input path/to/tennis_video.mp4 --output results/
```

### Command Line Arguments

- `--input` / `-i`: Input video file path (required)
- `--output` / `-o`: Output directory for results (default: `results`)
- `--model` / `-m`: YOLO model path (default: `yolov8x.pt`)

## Output Structure

The application generates the following output structure:

```
results/
├── court_detection.jpg          # Court detection visualization
├── birds_eye_view.jpg          # Bird's eye view transformation
├── analysis_summary.txt         # Human-readable summary
├── comprehensive_analysis_report.json  # Detailed JSON report
├── videos/
│   └── tracked_players.mp4     # Video with tracking annotations
├── plots/
│   └── player_movements.png    # Movement trajectory plots
└── data/
    └── tracking_data.json      # Raw tracking data
```

## Components

### 1. Court Detector (`court_detector.py`)

- **Purpose**: Detects tennis court boundaries and calculates homography
- **Methods**:
  - `detect_court_lines()`: Uses edge detection and Hough transform
  - `find_court_corners()`: Identifies court corner points
  - `validate_court_geometry()`: Validates detected court shape
  - `get_court_transform_matrix()`: Calculates homography matrix

### 2. Player Tracker (`player_tracker.py`)

- **Purpose**: Tracks players across video frames
- **Features**:
  - Multi-object tracking with unique IDs
  - Velocity and distance calculation
  - Track visualization and statistics
  - JSON export of tracking data

### 3. Tennis Analyzer (`tennis_analyzer.py`)

- **Purpose**: Main analysis pipeline
- **Capabilities**:
  - Integrates court detection and player tracking
  - Generates comprehensive reports
  - Creates visualization plots
  - Exports analysis results

### 4. Main Application (`main.py`)

- **Purpose**: Command-line interface and orchestration
- **Features**:
  - Argument parsing and validation
  - Complete analysis pipeline
  - Error handling and progress reporting

## Analysis Results

### Court Detection

- **Court Corners**: 4 keypoints defining court boundaries
- **Homography Matrix**: 3x3 transformation matrix for bird's eye view
- **Validation**: Geometric validation of detected court shape

### Player Tracking

- **Track IDs**: Unique identifiers for each detected player
- **Positions**: Time-series of player positions
- **Velocities**: Movement velocity vectors
- **Distances**: Cumulative distance covered by each player

### Distance Analysis

- **Total Distance**: Sum of all movements for each player
- **Average Speed**: Mean velocity over tracking period
- **Movement Patterns**: Trajectory analysis and visualization

## Example Output

The application generates comprehensive analysis reports including player tracking data, distance calculations, and movement visualizations.

## Technical Details

### Court Detection Algorithm

1. **Edge Detection**: Canny edge detector applied to grayscale frame
2. **Line Detection**: Hough line transform identifies straight lines
3. **Line Filtering**: Separates horizontal and vertical lines
4. **Corner Detection**: Finds intersections of horizontal and vertical lines
5. **Geometry Validation**: Validates aspect ratio and shape

### Player Tracking Algorithm

1. **Detection**: YOLO model detects persons in each frame
2. **Association**: Hungarian algorithm matches detections to existing tracks
3. **Prediction**: Kalman filter predicts next positions
4. **Update**: Updates track with new detections
5. **Termination**: Removes tracks that haven't been updated recently

### Distance Calculation

- **Euclidean Distance**: Calculated between consecutive positions
- **Cumulative Sum**: Total distance accumulated over time
- **Speed Calculation**: Distance per frame converted to m/s

## Performance Considerations

### Optimization Tips

1. **GPU Acceleration**: Use CUDA-compatible GPU for faster YOLO inference
2. **Video Resolution**: Lower resolution videos process faster
3. **Model Size**: Use smaller YOLO models (e.g., `yolov8n.pt`) for speed
4. **Batch Processing**: Process multiple videos in sequence

### Memory Usage

- **YOLO Model**: ~200MB for YOLOv8x
- **Video Processing**: Depends on video resolution and length
- **Tracking Data**: Minimal memory footprint for track storage

## Troubleshooting

### Common Issues

1. **Court Not Detected**

   - Ensure video has clear court lines
   - Check video quality and lighting

2. **Players Not Tracked**

   - Verify YOLO model is downloaded
   - Ensure players are visible in video

3. **Poor Distance Accuracy**
   - Use high-resolution videos
   - Ensure stable camera position
