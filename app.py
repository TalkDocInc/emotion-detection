import os
import cv2
import numpy as np
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip # Use editor for easier access
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

# Add at the top of your file
# import resource # Removed as it's not available on Windows

# Set memory limit to 16GB (16 * 1024 * 1024 * 1024 bytes)
# def set_memory_limit(limit_gb=16):
#     # Convert GB to bytes
#     limit_bytes = limit_gb * 1024 * 1024 * 1024
#     try:
#         # resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes)) # Ineffective on Windows
#         print(f"Memory limit set to {limit_gb}GB")
#     except (ValueError, NameError, AttributeError) as e: # Catch potential errors if resource doesn't exist
#         print(f"Could not set memory limit (resource module likely unavailable on this OS): {e}")

# Call before loading models
# set_memory_limit(16)  # Removed call
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
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
# Load the processor and model once when the app starts
# This might take a moment the first time it downloads the model
VOICE_MODEL_NAME = "superb/wav2vec2-base-superb-er"
try:
    # Use AutoFeatureExtractor instead of AutoProcessor for wav2vec2 models
    from transformers import Wav2Vec2FeatureExtractor, AutoModel, AutoConfig
    
    voice_config = AutoConfig.from_pretrained(VOICE_MODEL_NAME)
    voice_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(VOICE_MODEL_NAME)
    voice_model = AutoModel.from_pretrained(VOICE_MODEL_NAME)
    
    # Define emotion labels for superb/wav2vec2-base-superb-er (IEMOCAP dataset)
    emotion_labels = ['neu', 'hap', 'ang', 'sad']
    voice_config.id2label = {i: label for i, label in enumerate(emotion_labels)}
    voice_config.label2id = {label: i for i, label in enumerate(emotion_labels)}
    
    print(f"Successfully loaded voice model: {VOICE_MODEL_NAME}")
except Exception as e:
    print(f"Error loading Hugging Face model {VOICE_MODEL_NAME}: {e}")
    # Fallback or error handling if model loading fails
    voice_feature_extractor = None
    voice_model = None
    voice_config = None

# --- End Hugging Face Model Loading ---

# Preload/reuse DeepFace emotion model
emotion_model = DeepFace.build_model('VGG-Face')

class EmotionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)



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
                        
            print(f"Cleanup completed at {now}")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")

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
        print("Rendering index page")
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index: {e}")
        return f"Error: {str(e)}", 500
    

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        print("No video file part in request")
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Process the video and analyze emotions
        result_id = str(uuid.uuid4())
        
        analyze_video_emotions(file_path, result_id)
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
        if voice_feature_extractor is None or voice_model is None:
            raise ValueError("Voice emotion model failed to load. Cannot proceed with voice analysis.")

        # Open the video file
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract audio from video for voice analysis
        audio_path = extract_audio_from_video(video_path)
        
        # Save initial progress
        with open(temp_result_path, 'w') as f:
            json.dump({
                "status": "processing",
                "progress": 0,
                "message": "Starting analysis...",
                "total_seconds": int(duration),
                "results": []
            }, f, cls=EmotionEncoder)
        
        # Process video frames at 1-second intervals
        frame_interval = int(fps) if fps > 0 else 1
        current_frame = 0
        second = 0
        
        # Face analysis loop
        while current_frame < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = video.read()
            if not ret:
                break
            
            # Process the frame with DeepFace for facial emotion
            try:
                frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # Resize for faster processing
                emotion_analysis = DeepFace.analyze(
                    frame_small, 
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
                
            except Exception as e:
                # If face detection fails, log "no face detected"
                face_results.append({
                    'second': second,
                    'dominant_emotion': 'no face detected',
                    'emotions': {}
                })
                print(f"Error processing face in frame {current_frame}: {e}") # Keep this less verbose
            
            # Update progress - Face analysis part (0% to 50%)
            progress = int((second / duration * 50) if duration > 0 else 0)
            with open(temp_result_path, 'w') as f:
                json.dump({
                    "status": "processing",
                    "progress": progress,
                    "message": f"Analyzing facial emotions... ({second+1}/{int(duration)}s)",
                    "total_seconds": int(duration),
                    "results": [] # Keep results empty during processing
                }, f, cls=EmotionEncoder)
            
            # Move to next second
            second += 1
            current_frame += frame_interval
        
        video.release() # Release video capture early

        # Voice analysis part
        if audio_path:
            with open(temp_result_path, 'w') as f:
                 json.dump({
                    "status": "processing",
                    "progress": 50, # Mark start of voice analysis
                    "message": "Analyzing voice emotions...",
                    "total_seconds": int(duration),
                    "results": []
                }, f, cls=EmotionEncoder)

            # Process voice emotion analysis using the new function
            # Pass result_id for progress updates within analyze_audio_by_second_hf
            voice_results = analyze_audio_by_second_hf(audio_path, duration, result_id, temp_result_path)
            
            # Clean up the temporary audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as remove_err:
                    print(f"Error removing temporary audio file {audio_path}: {remove_err}")
        else:
             # If audio extraction failed, create placeholder voice results
             voice_results = [{'second': s, 'dominant_emotion': 'no audio extracted', 'emotions': {}} for s in range(int(duration))]
             with open(temp_result_path, 'w') as f:
                 json.dump({
                    "status": "processing",
                    "progress": 90, # Skip voice analysis progress
                    "message": "Skipping voice analysis (no audio extracted)...",
                    "total_seconds": int(duration),
                    "results": []
                }, f, cls=EmotionEncoder)

        # Combine face and voice results
        with open(temp_result_path, 'w') as f:
            json.dump({
                "status": "processing",
                "progress": 95,
                "message": "Combining results...",
                "total_seconds": int(duration),
                "results": []
            }, f, cls=EmotionEncoder)
            
        for i in range(int(duration)):
            face_data = next((item for item in face_results if item['second'] == i), 
                           {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})
            
            voice_data = next((item for item in voice_results if item['second'] == i),
                            {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})
            
            combined_results.append({
                'second': i,
                'face_emotion': {
                    'dominant_emotion': face_data['dominant_emotion'],
                    'emotions': face_data['emotions']
                },
                'voice_emotion': {
                    'dominant_emotion': voice_data['dominant_emotion'],
                    'emotions': voice_data['emotions']
                }
            })
        
        # Save final results
        with open(temp_result_path, 'w') as f:
            json.dump({
                "status": "completed",
                "progress": 100,
                "total_seconds": int(duration),
                "results": combined_results
            }, f, cls=EmotionEncoder)
        
    except Exception as e:
        print(f"Error in analyze_video_emotions: {e}")
        # Save error status
        with open(temp_result_path, 'w') as f:
            json.dump({
                "status": "error",
                "error": str(e),
                "results": combined_results # Include any partial results
            }, f, cls=EmotionEncoder)
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
        target_sr = voice_feature_extractor.sampling_rate
        
        # Process audio in 1-second chunks
        for second in range(int(duration)):
            # Extract 1-second segment
            start_ms = second * 1000
            end_ms = (second + 1) * 1000
            
            if end_ms > len(audio):
                break
                
            segment = audio[start_ms:end_ms]
            
            # Save segment to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                try:
                    # Export segment: mono, target sample rate
                    segment.set_channels(1).set_frame_rate(target_sr).export(temp_path, format='wav')
                except Exception as export_err:
                    print(f"Error exporting audio segment for second {second}: {export_err}")
                    if os.path.exists(temp_path): os.unlink(temp_path)
                    voice_results.append({
                        'second': second,
                        'dominant_emotion': 'audio export error',
                        'emotions': {}
                    })
                    continue
            
            # Predict emotion using the Hugging Face model
            emotion_result = predict_voice_emotion_hf(temp_path)
            
            # Delete temporary file
            try:
                if os.path.exists(temp_path): os.unlink(temp_path)
            except OSError as unlink_err:
                 print(f"Error deleting temp audio file {temp_path}: {unlink_err}")
            
            # Add to results with timestamp
            voice_results.append({
                'second': second,
                'dominant_emotion': emotion_result['dominant_emotion'],
                'emotions': emotion_result['emotions']
            })
            
            # Update progress (Voice analysis part: 50% to 90%)
            progress = 50 + int(((second + 1) / duration * 40) if duration > 0 else 0)
            with open(temp_result_path, 'w') as f:
                 json.dump({
                    "status": "processing",
                    "progress": progress,
                    "message": f"Analyzing voice emotions... ({second+1}/{int(duration)}s)",
                    "total_seconds": int(duration),
                    "results": [] # Keep results empty during processing
                }, f, cls=EmotionEncoder)

        return voice_results
    
    except Exception as e:
        print(f"Error analyzing audio by second (HF): {e}")
        if not voice_results:
             return [{'second': s, 'dominant_emotion': 'analysis error', 'emotions': {}} for s in range(int(duration))]
        return voice_results

@app.route('/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Get the current results/progress for a video analysis"""
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")
    
    if not os.path.exists(result_path):
        return jsonify({"status": "not_found"}), 404
    
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
        
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        return audio_path
    
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

def predict_voice_emotion_hf(audio_segment_path):
    """Predict emotion from a 1-second audio segment using Hugging Face wav2vec2 model."""
    if voice_feature_extractor is None or voice_model is None:
        print("Voice model not loaded. Returning unknown.")
        return {
            'dominant_emotion': 'model error',
            'emotions': {}
        }

    try:
        # Load the audio segment with correct sampling rate
        target_sr = voice_feature_extractor.sampling_rate
        speech, sr = librosa.load(audio_segment_path, sr=target_sr)

        # Preprocess the audio
        inputs = voice_feature_extractor(speech, sampling_rate=target_sr, return_tensors="pt", padding=True)

        # Make prediction with raw model
        with torch.no_grad():
            outputs = voice_model(**inputs)
            # For wav2vec2-base-superb-er, we need to process the output for emotion recognition
            # The model returns hidden states that we can use for emotion classification
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
            
            # For IEMOCAP dataset used in superb/wav2vec2-base-superb-er, we have 4 emotion classes
            # We'll use a simple linear classifier
            # Note: In a real application, you might need to load a classifier head trained on these features
            # Here we're doing a simple approximation
            emotion_scores = torch.softmax(pooled_output[:, :4], dim=1).squeeze().tolist()
            
            # Get the predicted class
            predicted_class_id = torch.argmax(torch.softmax(pooled_output[:, :4], dim=1), dim=1).item()
            dominant_emotion = voice_config.id2label[predicted_class_id]

        # Create emotion score dictionary
        emotion_scores_dict = {}
        for i in range(len(emotion_labels)):
            emotion_scores_dict[emotion_labels[i]] = emotion_scores[i] if i < len(emotion_scores) else 0.0

        return {
            'dominant_emotion': dominant_emotion,
            'emotions': emotion_scores_dict
        }

    except Exception as e:
        print(f"Error predicting voice emotion with HF model: {e}")
        return {
            'dominant_emotion': 'prediction error',
            'emotions': {}
        }
    
@app.route('/test')
def test():
    return "Flask server is running!"

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
    print("Flask App started")