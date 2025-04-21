import os
import cv2
import numpy as np
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
from deepface import DeepFace
import time
import json
import threading
import shutil
from datetime import datetime, timedelta
import librosa
import soundfile as sf
from pydub import AudioSegment
import joblib
from sklearn.preprocessing import StandardScaler
import tempfile
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import logging
import logging_config  # initialize file logging (rotating file handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload size
app.config['CLEANUP_INTERVAL'] = 24  # hours - files older than this will be removed

# Create the uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Create a models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Limit TensorFlow threads, disable XLA
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# --- Hugging Face Model Loading ---
VOICE_MODEL_NAME = "superb/wav2vec2-base-superb-er"
LOCAL_MODEL_DIR = os.path.join('models', 'wav2vec2-er')
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Define a function to load the model with retries
def load_voice_model(model_name, local_dir, max_retries=3):
    """Load the voice emotion model with retries and local caching"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading voice model (attempt {attempt+1}/{max_retries})...")
            
            # Try loading from local directory first if it exists
            if os.path.exists(os.path.join(local_dir, "config.json")):
                logger.info(f"Found local model at {local_dir}, loading...")
                processor = AutoProcessor.from_pretrained(local_dir)
                model = AutoModelForAudioClassification.from_pretrained(local_dir)
                logger.info("Successfully loaded voice model from local cache")
                return processor, model
                
            # Otherwise, download from Hugging Face and cache locally
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(model_name)
            
            # Save the model and processor locally for future offline use
            logger.info(f"Saving model to {local_dir} for future offline use")
            processor.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
            
            logger.info(f"Successfully loaded audio classification model: {model_name}")
            return processor, model
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, etc.
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.exception(f"All {max_retries} attempts to load voice model failed")
                return None, None

# Load the model using our new function
voice_processor, voice_model = load_voice_model(VOICE_MODEL_NAME, LOCAL_MODEL_DIR)
# Get emotion labels if model loaded successfully
emotion_labels = list(voice_model.config.id2label.values()) if voice_model is not None else []
# --- End Hugging Face Model Loading ---

# --- Depression Analysis Configuration ---
# Depression indicators in facial emotions (0-1 scale, higher means more associated with depression)
FACE_DEPRESSION_WEIGHTS = {
    'angry': 0.6,
    'disgust': 0.5,
    'fear': 0.7,
    'happy': -0.8,  # Negative because happiness indicates less depression
    'sad': 0.9,
    'surprise': 0.0,
    'neutral': 0.0,
    'no face detected': 0.0  # No contribution if no face detected
}

# Depression indicators in voice emotions (0-1 scale)
VOICE_DEPRESSION_WEIGHTS = {
    'neu': 0.0,  
    'hap': -0.8,  # Happiness in voice suggests less depression
    'ang': 0.5,   # Anger can be associated with depression
    'sad': 0.9,   # Sadness strongly correlated with depression
    'no audio extracted': 0.0,
    'audio export error': 0.0,
    'model error': 0.0,
    'prediction error': 0.0,
    'analysis error': 0.0,
    'no data': 0.0
}

# Overall weighting between face and voice analysis for depression score
FACE_WEIGHT = 0.6  # Face analysis contributes 60% to the depression score
VOICE_WEIGHT = 0.4  # Voice analysis contributes 40% to the depression score
# --- End Depression Analysis Configuration ---

class EmotionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

def calculate_depression_score(face_emotions, voice_emotions):
    """
    Calculate a depression score based on face and voice emotion analysis.
    
    Args:
        face_emotions: Dict of facial emotion scores or dominant emotion string
        voice_emotions: Dict of voice emotion scores or dominant emotion string
    
    Returns:
        A score between 0-100 where higher numbers indicate higher likelihood of depression
    """
    face_score = 0.0
    voice_score = 0.0
    
    # Calculate face depression contribution
    if isinstance(face_emotions, dict) and face_emotions:
        # If we have detailed emotion scores
        for emotion, score in face_emotions.items():
            if emotion in FACE_DEPRESSION_WEIGHTS:
                face_score += score * FACE_DEPRESSION_WEIGHTS[emotion]
    elif isinstance(face_emotions, str):
        # If we just have a dominant emotion string
        if face_emotions in FACE_DEPRESSION_WEIGHTS:
            face_score = FACE_DEPRESSION_WEIGHTS[face_emotions]
    
    # Calculate voice depression contribution
    if isinstance(voice_emotions, dict) and voice_emotions:
        # If we have detailed emotion scores
        for emotion, score in voice_emotions.items():
            if emotion in VOICE_DEPRESSION_WEIGHTS:
                voice_score += score * VOICE_DEPRESSION_WEIGHTS[emotion]
    elif isinstance(voice_emotions, str):
        # If we just have a dominant emotion string
        if voice_emotions in VOICE_DEPRESSION_WEIGHTS:
            voice_score = VOICE_DEPRESSION_WEIGHTS[voice_emotions]
    
    # Combine scores with appropriate weighting
    combined_score = (face_score * FACE_WEIGHT) + (voice_score * VOICE_WEIGHT)
    
    # Scale to 0-100 range (assuming original scores are roughly -1 to 1)
    scaled_score = (combined_score + 1) * 50
    
    # Ensure score is within 0-100 bounds
    final_score = max(0, min(100, scaled_score))
    
    return final_score

# Lock and helper for atomic progress file writes
dump_lock = threading.Lock()
def write_progress(result_path, payload):
    """Atomically write JSON payload to result file with lock"""
    temp_path = f"{result_path}.tmp"
    with dump_lock:
        with open(temp_path, 'w') as f:
            json.dump(payload, f, cls=EmotionEncoder)
        os.replace(temp_path, result_path)

# Start the cleanup thread
def cleanup_old_files():
    """
    Periodically clean up old files in the uploads directory
    to prevent disk space issues.
    """
    while True:
        try:
            # Sleep for a few hours
            time.sleep(60 * 60 * 6)  # Clean every 6 hours
            
            now = datetime.now()
            cleanup_before = now - timedelta(hours=app.config['CLEANUP_INTERVAL'])
            
            # Check all files in the uploads directory
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Get the file's modification time
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # If the file is older than the cleanup interval, delete it
                if file_mod_time < cleanup_before:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        
            logger.info(f"Cleanup completed at {now}")
                
        except Exception:
            logger.exception("Error during cleanup")

# Start the cleanup thread when the app starts
cleanup_thread = threading.Thread(target=cleanup_old_files)
cleanup_thread.daemon = True
cleanup_thread.start()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    try:
        logger.info("Rendering index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return f"Error: {str(e)}", 500
    

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        logger.warning("No video file part in request")
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Process the video and analyze emotions asynchronously
        result_id = str(uuid.uuid4())
        # Start background thread for analysis so request returns immediately
        threading.Thread(target=analyze_video_emotions, args=(file_path, result_id), daemon=True).start()
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': unique_filename,
            'result_id': result_id
        })

    else:
        return jsonify({'error': 'File type not allowed'}), 400

def analyze_video_emotions(video_path, result_id):
    """Analyze emotions in the video frame by frame with both face and voice analysis"""
    face_results = []
    voice_results = []
    combined_results = []
    temp_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")
    
    try:
        # Check if voice model is loaded
        if voice_processor is None or voice_model is None:
            raise ValueError("Voice emotion model failed to load. Cannot proceed with voice analysis.")

        # Open the video file
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract audio from video for voice analysis
        audio_path = extract_audio_from_video(video_path)
        
        # Save initial progress
        write_progress(temp_result_path, {
            "status": "processing",
            "progress": 0,
            "message": "Starting analysis...",
            "total_seconds": int(duration),
            "results": []
        })
        
        # Process video frames at 1-second intervals with accurate time mapping
        num_seconds = int(duration)
        for second in range(num_seconds):
            frame_index = round(second * fps)
            if frame_index >= total_frames:
                break
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video.read()
            if not ret:
                continue
            
            # Process the frame with DeepFace for facial emotion
            try:
                emotion_analysis = DeepFace.analyze(
                    frame, 
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=False                )
                
                # Get the dominant emotion
                if isinstance(emotion_analysis, list):
                    emotion_data = emotion_analysis[0]
                else:
                    emotion_data = emotion_analysis
                
                dominant_emotion = emotion_data['dominant_emotion']
                emotion_scores = emotion_data['emotion']
                
                # Add to face results
                face_results.append({
                    'second': second,
                    'dominant_emotion': dominant_emotion,
                    'emotions': emotion_scores
                })
                
            except Exception:
                # If face detection fails, log "no face detected"
                face_results.append({
                    'second': second,
                    'dominant_emotion': 'no face detected',
                    'emotions': {}
                })
                logger.warning(f"Error processing face in frame {frame_index}", exc_info=True)
            
            # Update progress - Face analysis part (0% to 50%)
            progress = int((second / duration * 50) if duration > 0 else 0)
            write_progress(temp_result_path, {
                "status": "processing",
                "progress": progress,
                "message": f"Analyzing facial emotions... ({second+1}/{int(duration)}s)",
                "total_seconds": int(duration),
                "results": [] # Keep results empty during processing
            })
        
        video.release() # Release video capture early

        # Voice analysis part
        if audio_path:
            write_progress(temp_result_path, {
                "status": "processing",
                "progress": 50, # Mark start of voice analysis
                "message": "Analyzing voice emotions...",
                "total_seconds": int(duration),
                "results": []
            })

            # Process voice emotion analysis using the new function
            # Pass result_id for progress updates within analyze_audio_by_second_hf
            voice_results = analyze_audio_by_second_hf(audio_path, duration, result_id, temp_result_path)
            
            # Clean up the temporary audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as remove_err:
                    logger.warning(f"Error removing temporary audio file {audio_path}: {remove_err}")
        else:
             # If audio extraction failed, create placeholder voice results
             voice_results = [{'second': s, 'dominant_emotion': 'no audio extracted', 'emotions': {}} for s in range(int(duration))]
             write_progress(temp_result_path, {
                "status": "processing",
                "progress": 90, # Skip voice analysis progress
                "message": "Skipping voice analysis (no audio extracted)...",
                "total_seconds": int(duration),
                "results": []
            })

        # Combine face and voice results
        write_progress(temp_result_path, {
            "status": "processing",
            "progress": 95,
            "message": "Combining results and calculating depression scores...",
            "total_seconds": int(duration),
            "results": []
        })
            
        # Calculate overall depression score
        total_depression_score = 0
        depression_scores_by_second = []
            
        for i in range(int(duration)):
            face_data = next((item for item in face_results if item['second'] == i), 
                           {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})
            
            voice_data = next((item for item in voice_results if item['second'] == i),
                            {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})
            
            # Calculate depression score for this second
            depression_score = calculate_depression_score(
                face_data['emotions'] if face_data['emotions'] else face_data['dominant_emotion'],
                voice_data['emotions'] if voice_data['emotions'] else voice_data['dominant_emotion']
            )
            
            # Store the depression score for this second
            depression_scores_by_second.append(depression_score)
            
            # Add to the running total for the overall score
            total_depression_score += depression_score
            
            combined_results.append({
                'second': i,
                'face_emotion': {
                    'dominant_emotion': face_data['dominant_emotion'],
                    'emotions': face_data['emotions']
                },
                'voice_emotion': {
                    'dominant_emotion': voice_data['dominant_emotion'],
                    'emotions': voice_data['emotions']
                },
                'depression_score': depression_score
            })
        
        # Calculate overall depression score (average of all seconds)
        overall_depression_score = total_depression_score / len(combined_results) if combined_results else 0
        
        # Save final results
        write_progress(temp_result_path, {
            "status": "completed",
            "progress": 100,
            "total_seconds": int(duration),
            "overall_depression_score": overall_depression_score,
            "results": combined_results
        })
        
    except Exception as e:
        logger.exception("Error in analyze_video_emotions")
        # Save error status
        write_progress(temp_result_path, {
            "status": "error",
            "error": str(e),
            "results": combined_results # Include any partial results
        })
    finally:
        # Ensure video capture is released if an error occurred before release
        if 'video' in locals() and video.isOpened():
            video.release()
        # Clean up audio path if it exists and wasn't cleaned up
        if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
             try:
                 os.remove(audio_path)
             except OSError:
                 pass # Ignore error if already removed or removal fails

# New function for HF model analysis, replacing the old analyze_audio_by_second
def analyze_audio_by_second_hf(audio_path, duration, result_id, temp_result_path):
    """Analyze audio emotion second by second using Hugging Face model with progress updates."""
    voice_results = []
    
    try:
        # Load full audio using pydub for easy slicing
        audio = AudioSegment.from_file(audio_path)
        target_sr = voice_processor.sampling_rate
        
        # Process audio in 1-second chunks in-memory (no temp files)
        for second in range(int(duration)):
            # Extract 1-second segment
            start_ms = second * 1000
            end_ms = (second + 1) * 1000
            if end_ms > len(audio):
                break
            segment = audio[start_ms:end_ms]
            # Convert to mono and target sample rate
            segment = segment.set_channels(1).set_frame_rate(target_sr)
            # Convert to numpy array and normalize
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            max_val = float(2 ** (8 * segment.sample_width - 1))
            speech = samples / max_val
            # Predict emotion using the Hugging Face model on raw audio
            emotion_result = predict_voice_emotion_hf(speech=speech, sr=target_sr)
            # Add to results with timestamp
            voice_results.append({
                'second': second,
                'dominant_emotion': emotion_result['dominant_emotion'],
                'emotions': emotion_result['emotions']
            })
            
            # Update progress (Voice analysis part: 50% to 90%)
            progress = 50 + int(((second + 1) / duration * 40) if duration > 0 else 0)
            write_progress(temp_result_path, {
                "status": "processing",
                "progress": progress,
                "message": f"Analyzing voice emotions... ({second+1}/{int(duration)}s)",
                "total_seconds": int(duration),
                "results": [] # Keep results empty during processing
            })

        return voice_results
    
    except Exception:
        logger.exception("Error analyzing audio by second (HF)")
        if not voice_results:
             return [{'second': s, 'dominant_emotion': 'analysis error', 'emotions': {}} for s in range(int(duration))]
        return voice_results

@app.route('/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Get the current results/progress for a video analysis"""
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")
    
    if not os.path.exists(result_path):
        return jsonify({"status": "not_found"}), 200
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def extract_audio_from_video(video_path, output_dir=None):
    """Extract audio from video file"""
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    try:
        # Generate audio filename
        base_name = os.path.basename(video_path)
        audio_name = f"{os.path.splitext(base_name)[0]}_audio.wav"
        audio_path = os.path.join(output_dir, audio_name)
        
        # Extract audio using moviepy with context manager for automatic close
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        return audio_path
    except Exception:
        logger.exception("Error extracting audio from video")
        return None

def predict_voice_emotion_hf(audio_segment_path=None, speech=None, sr=None):
    """Predict emotion from a 1-second audio segment using Hugging Face wav2vec2 model.
    Can accept raw audio array (`speech` and `sr`) or a filesystem path."""
    if voice_processor is None or voice_model is None:
        logger.warning("Voice model not loaded. Returning unknown.")
        return {
            'dominant_emotion': 'model error',
            'emotions': {}
        }
    try:
        # Load or use provided raw audio
        if speech is None:
            speech, sr = librosa.load(audio_segment_path, sr=voice_processor.sampling_rate)
        # Preprocess with classifier processor
        inputs = voice_processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = voice_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze()
            # Ensure list format
            probs = probs.tolist() if hasattr(probs, 'tolist') else [float(probs)]
            predicted_id = int(torch.argmax(logits, dim=1).item())
            dominant_emotion = voice_model.config.id2label[predicted_id]
        # Map probabilities to labels
        emotion_scores_dict = {label: probs[idx] for idx, label in voice_model.config.id2label.items()}

        return {'dominant_emotion': dominant_emotion, 'emotions': emotion_scores_dict}
    except Exception:
        logger.exception("Error predicting voice emotion")
        return {
            'dominant_emotion': 'prediction error',
            'emotions': {}
        }

@app.route('/test')
def test():
    return "Flask server is running!"

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)
    logger.info("Flask App started")