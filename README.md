# Emotion Detection on Video Footage

A web application that analyzes emotions in uploaded video footage on a second-by-second basis.

## Features

- Upload video files (MP4, AVI, MOV, WMV, MKV)
- Real-time progress tracking of video analysis
- Second-by-second emotion detection
- Visual representation of emotion distribution
- Detailed emotion scores for each second of video

## Disclaimer

**This application is for informational and entertainment purposes only. It is not a medical tool and should not be used to diagnose or assess mental health conditions, including depression. The emotion detection is based on algorithmic analysis of facial expressions and may not accurately reflect a person's true emotional state or mental health.**

## Technology Stack

- **Backend**: Python with Flask
- **Frontend**: HTML, CSS, JavaScript
- **Emotion Detection**: DeepFace library with pre-trained models
- **Video Processing**: OpenCV and MoviePy

## Setup Instructions

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Emotion-Detection-on-Video-Footage
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open a web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## How It Works

1. User uploads a video file through the web interface
2. The backend processes the video frame by frame at 1-second intervals
3. Each frame is analyzed using DeepFace to detect emotions
4. Results are stored and provided to the frontend
5. The frontend displays emotions detected over time with visualizations

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Requirements

- Python 3.7+
- Sufficient memory to process video files
- Webcam for local testing (optional)

## License

MIT

## Disclaimer

This code is not designed to diagnose depression.
