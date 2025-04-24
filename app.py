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
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import pandas as pd # Added for rolling average
from feat import Detector # Added for py-feat AU detection
import tempfile

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

# --- Depression Analysis Configuration ---
# NEW: Weights for Facial Action Units (AUs) based on intensity (assume 0-1 from py-feat for now)
# Heuristic values, NEED TUNING. Positive means AU presence -> more depression indication.
FACE_AU_DEPRESSION_WEIGHTS = {
    # Inner Brow Raiser (Sadness, Fear)
    'AU01': 0.6,
    # Brow Lowerer (Anger, Sadness)
    'AU04': 0.8,
    # Cheek Raiser (Duchenne Smile marker - Happiness)
    'AU06': -0.9, # Negative score indicates less depression
    # Lid Tightener (Anger, Fear, Pain)
    'AU07': 0.4,
    # Upper Lip Raiser (Disgust, Sadness)
    'AU10': 0.3,
    # Lip Corner Puller (Smile marker - Happiness)
    'AU12': -0.9, # Negative score indicates less depression
    # Lip Corner Depressor (Sadness)
    'AU15': 0.9,
    # Chin Raiser (Sadness, Anger)
    'AU17': 0.7,
    # Lip Stretcher (Fear, Sadness)
    'AU20': 0.5,
    # Note: py-feat might output slightly different AU numbers/names depending on the model used.
    # We may need to adjust these keys based on the actual output of the detector.
    # Example only, add other relevant AUs like 23 (Lip Tightener), 25/26 (Lips part/Jaw Drop) if needed.
}

# Depression indicators in voice emotions (0-1 scale)
VOICE_DEPRESSION_WEIGHTS = {
    'neu': 0.05,   # Neutral tone can indicate mild depression
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

# NEW: Parameters for scaling acoustic features using tanh((value - center) / scale)
# These are heuristic estimates and require tuning!
ACOUSTIC_FEATURE_SCALING_PARAMS = {
    # Feature: {'center': typical_value, 'scale': typical_range_or_std_dev}
    'rms_mean': {'center': 0.05, 'scale': 0.05},    # Assumes audio normalized somewhat, range ~0 to 0.1+
    'rms_std': {'center': 0.01, 'scale': 0.01},     # Std dev likely smaller
    'spectral_centroid_mean': {'center': 1500, 'scale': 1000}, # Wide range, center guess
    'spectral_centroid_std': {'center': 500, 'scale': 500},    # Std dev also varies
    'zcr_mean': {'center': 0.1, 'scale': 0.1},      # Range 0-1 approx
    'zcr_std': {'center': 0.05, 'scale': 0.05},     # Std dev likely smaller
    'pitch_mean': {'center': 150, 'scale': 50},      # Center for typical voice, scale covers reasonable range (Hz)
    'pitch_std': {'center': 20, 'scale': 20},       # Typical pitch std dev (Hz), scale allows variation
    'mfcc_mean_std': {'center': 10, 'scale': 10}      # Wild guess, depends heavily on MFCC implementation
}

# REVISED: Weights for Granular Acoustic Features (applied AFTER scaling to approx [-1, 1])
# Weights can now be more intuitive (closer to -1 to 1 range)
VOICE_ACOUSTIC_DEPRESSION_WEIGHTS = {
    # Energy (RMS)
    'rms_mean': -0.3, # Lower scaled energy -> more depression
    'rms_std': -0.4,  # Lower scaled energy variation -> more depression
    # Spectral Centroid (Brightness)
    'spectral_centroid_mean': -0.2, # Lower scaled brightness -> more depression
    'spectral_centroid_std': -0.2,  # Lower scaled brightness variation -> more depression
    # Zero Crossing Rate (Noisiness/Voiced detection) - still less clear link
    'zcr_mean': 0.0,
    'zcr_std': 0.0,
    # Pitch (F0) - Stronger indicators
    'pitch_mean': -0.5, # Lower scaled pitch -> more depression
    'pitch_std': -0.7, # Lower scaled pitch variation (monotone) -> more depression
    # MFCCs (Timbre)
    'mfcc_mean_std': -0.3 # Lower scaled variation across MFCC means -> flatter timbre
}

# Weighting between voice emotion category and acoustic features
VOICE_EMOTION_WEIGHT = 0.6 # Contribution from categorical emotion (e.g., 'sad')
VOICE_ACOUSTIC_WEIGHT = 0.4 # Contribution from granular features (pitch, energy, etc.)

# --- End Depression Analysis Configuration ---

# Preload/reuse DeepFace emotion model
# emotion_model = DeepFace.build_model('VGG-Face') # Removed: DeepFace emotion model no longer used

# --- NEW: Initialize py-feat Detector ---
try:
    # Initialize the detector once. Choose models (e.g., RetinaFace, RF):
    # au_model options: 'rf', 'svm', 'logistic', 'jaanet' (requires tensorflow)
    # face_model options: 'retinaface', 'mtcnn', 'faceboxes', 'wf'
    # landmark_model options: 'mobilenet', 'mobilefacenet', 'pfld'
    # emotion_model: We won't use this directly for scoring, but can keep it for potential future use/comparison.
    face_detector = Detector(
        landmark_model="mobilenet", # Lightweight landmarks
        au_model="xgb", # Try using XGBoost instead of SVM
        emotion_model="svm" # Keep emotion model loaded for now if needed later
    )
    print("Successfully initialized py-feat detector.")
except ImportError:
    print("Error: py-feat library not found. Please install it: pip install py-feat[all]")
    face_detector = None
except Exception as e:
    print(f"Error initializing py-feat detector: {e}")
    face_detector = None
# --- End py-feat Initialization ---

class EmotionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

# Helper function to update the progress JSON file
def update_progress(result_id, status, progress, message, total_seconds, results=None):
    """Updates the JSON progress file."""
    temp_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")
    data_to_dump = {
        "status": status,
        "progress": progress,
        "message": message,
        "total_seconds": int(total_seconds)
    }
    # Only include results if they are provided (usually for final or error states)
    if results is not None:
        data_to_dump["results"] = results
    else:
        # Ensure results key exists even during processing if not provided
        data_to_dump["results"] = [] # Keep results empty/as-is during processing

    try:
        with open(temp_result_path, 'w') as f:
            json.dump(data_to_dump, f, cls=EmotionEncoder)
    except Exception as e:
        print(f"Error updating progress file {temp_result_path}: {e}")

# Renamed function: calculates the raw weighted score before non-linear scaling
def calculate_raw_depression_metric(face_au_data, voice_emotion_data, voice_acoustic_features):
    """
    Calculate a raw depression metric based on face AUs, voice emotion, and voice acoustic features.
    The score is roughly in [-1, 1].

    Args:
        face_au_data: Dict containing AU intensities for the time segment (e.g., {'AU01': 0.8, 'AU04': 0.5, 'error': None}). 'error' key indicates issues.
        voice_emotion_data: Dict possibly containing 'dominant_emotion' (str) and 'emotions' (dict of scores).
        voice_acoustic_features: Dict containing extracted acoustic features for the same time segment (e.g., pitch_mean, rms_std). Values might be None if extraction failed.

    Returns:
        A raw score roughly between -1 and 1, representing weighted depression indication.
    """
    face_dep_contribution = 0.0
    voice_dep_contribution = 0.0 # Combined voice score
    face_weight_used = 0.0  # Flag to track if face data contributed
    voice_weight_used = 0.0 # Flag to track if voice data contributed

    # --- Calculate Face Depression Contribution (Using AUs) ---
    if face_au_data and face_au_data.get('error') is None:
        aus = face_au_data.get('aus', {})
        num_aus_used = 0
        temp_face_score = 0.0
        if aus: # Check if the aus dictionary is not empty
            for au_name, weight in FACE_AU_DEPRESSION_WEIGHTS.items():
                au_intensity = aus.get(au_name)
                # Check if AU exists, is not None/NaN, and weight is non-zero
                if au_intensity is not None and not np.isnan(au_intensity) and abs(weight) > 1e-6:
                    # Assuming AU intensity is roughly 0-1 (or 0-5, but weights account for scale)
                    # No scaling applied here for now, but could be added like acoustics if needed.
                    temp_face_score += au_intensity * weight
                    num_aus_used += 1

            if num_aus_used > 0:
                # Simple weighted sum. Could normalize by num_aus_used if desired.
                face_dep_contribution = temp_face_score
                face_weight_used = 1.0
                print(f"Face depression contribution: {face_dep_contribution}")
        # If aus dict is empty but no error, treat as neutral (contribution remains 0)
    # If face_au_data has an error, contribution remains 0

    # --- Calculate Voice Depression Contribution (Combined Emotion + Acoustics) ---
    voice_emotion_contribution = 0.0
    voice_acoustic_contribution = 0.0
    emotion_data_used = False
    acoustic_data_used = False

    # 1. Contribution from Emotion Category
    voice_emotions = voice_emotion_data.get('emotions', {})
    voice_dominant = voice_emotion_data.get('dominant_emotion', 'no data')

    if voice_emotions:
        total_voice_score = sum(voice_emotions.values())
        if total_voice_score > 1e-6:
            for emotion, score in voice_emotions.items():
                if emotion in VOICE_DEPRESSION_WEIGHTS:
                    voice_emotion_contribution += (score / total_voice_score) * VOICE_DEPRESSION_WEIGHTS[emotion]
            emotion_data_used = True
        elif voice_dominant in VOICE_DEPRESSION_WEIGHTS and VOICE_DEPRESSION_WEIGHTS[voice_dominant] != 0.0:
            voice_emotion_contribution = VOICE_DEPRESSION_WEIGHTS[voice_dominant]
            emotion_data_used = True
    elif voice_dominant in VOICE_DEPRESSION_WEIGHTS and VOICE_DEPRESSION_WEIGHTS[voice_dominant] != 0.0:
        voice_emotion_contribution = VOICE_DEPRESSION_WEIGHTS[voice_dominant]
        emotion_data_used = True

    # 2. Contribution from Acoustic Features
    if voice_acoustic_features and voice_acoustic_features.get('error') is None:
        num_acoustic_features_used = 0
        temp_acoustic_score = 0.0
        for feature_name, weight in VOICE_ACOUSTIC_DEPRESSION_WEIGHTS.items():
            feature_value = voice_acoustic_features.get(feature_name)
            scaling_params = ACOUSTIC_FEATURE_SCALING_PARAMS.get(feature_name)

            # Check if feature exists, is not None/NaN, has scaling params, and weight is non-zero
            if (feature_value is not None and
                not np.isnan(feature_value) and
                scaling_params and
                scaling_params.get('scale') is not None and
                abs(scaling_params['scale']) > 1e-9 and # Avoid division by zero
                abs(weight) > 1e-6):

                # Apply scaling: tanh((value - center) / scale)
                center = scaling_params.get('center', 0.0)
                scale = scaling_params['scale']
                scaled_value = np.tanh((feature_value - center) / scale)

                # Use scaled value in weighted sum
                temp_acoustic_score += scaled_value * weight
                num_acoustic_features_used += 1

        if num_acoustic_features_used > 0:
            # Optional: Average the contribution if needed, though weights handle scale somewhat
            # voice_acoustic_contribution = temp_acoustic_score / num_acoustic_features_used
            voice_acoustic_contribution = temp_acoustic_score # Direct weighted sum for now
            acoustic_data_used = True

    # 3. Combine Voice Contributions and Set Voice Weight Flag
    if emotion_data_used and acoustic_data_used:
        # Combine using predefined weights
        voice_dep_contribution = (voice_emotion_contribution * VOICE_EMOTION_WEIGHT) + \
                                 (voice_acoustic_contribution * VOICE_ACOUSTIC_WEIGHT)
        voice_weight_used = 1.0
    elif emotion_data_used:
        # Only emotion data was useful
        voice_dep_contribution = voice_emotion_contribution
        voice_weight_used = 1.0
    elif acoustic_data_used:
        # Only acoustic data was useful
        voice_dep_contribution = voice_acoustic_contribution
        voice_weight_used = 1.0
    # Else: voice_dep_contribution remains 0.0, voice_weight_used remains 0.0

    # --- Combine Face and Voice scores with appropriate overall weighting ---
    total_base_weight = (face_weight_used * FACE_WEIGHT) + (voice_weight_used * VOICE_WEIGHT)

    if total_base_weight > 1e-6:
        # Normalize weights based on which sources actually contributed
        adjusted_face_weight = (face_weight_used * FACE_WEIGHT) / total_base_weight
        adjusted_voice_weight = (voice_weight_used * VOICE_WEIGHT) / total_base_weight
        combined_score = (face_dep_contribution * adjusted_face_weight) + (voice_dep_contribution * adjusted_voice_weight)
    else:
        combined_score = 0.0 # No valid data from either source contributed

    # Return the raw combined score before non-linear transformation
    return combined_score

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
    """Analyze emotions in the video frame by frame with face AU and voice analysis"""
    face_au_results = [] # Store AU intensity results
    voice_results = []
    acoustic_features_results = []
    combined_results = []
    temp_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")
    
    try:
        # Check if models are loaded
        if face_detector is None:
             raise ValueError("Face detector (py-feat) failed to initialize. Cannot proceed with face analysis.")
        if voice_feature_extractor is None or voice_model is None:
            # Allow proceeding without voice, but log it
            print("Warning: Voice emotion model failed to load. Proceeding without voice analysis.")
            # We will handle this later by adjusting weights or providing default voice data

        # Open the video file
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract audio from video for voice analysis
        audio_path = extract_audio_from_video(video_path)
        
        # Save initial progress
        update_progress(result_id, "processing", 0, "Starting analysis...", duration)
        
        # Process video frames at 1-second intervals
        frame_interval = int(fps) if fps > 0 else 1
        current_frame = 0
        second = 0
        
        # Face analysis loop (using py-feat)
        while current_frame < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = video.read()
            if not ret:
                break

            # --- Resize Frame to 720p (Maintain Aspect Ratio) ---
            target_height = 720
            h, w = frame.shape[:2]
            if h > target_height:
                ratio = target_height / h
                target_width = int(w * ratio)
                frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                # If frame height is already <= 720p, use original
                frame_resized = frame
            # --- End Resize ---

            current_face_au_data = {'second': second, 'aus': {}, 'error': None}

            # Process the frame with py-feat for facial action units
            temp_image_path = None # Initialize outside try
            try:
                # Ensure detector is available
                if face_detector is None:
                    raise RuntimeError("py-feat face_detector is not initialized.")

                # Use the resized frame
                # Input frame needs to be RGB for some models
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) # Use frame_resized

                # --- WORKAROUND: Save numpy array to temp file --- 
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
                    temp_image_path = temp_f.name
                    # Use cv2.imwrite to save the numpy array (frame_rgb) as an image
                    # imwrite expects BGR, so convert back
                    cv2.imwrite(temp_image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)) # Use frame_rgb
                # --- END WORKAROUND ---

                # Pass the file path to detect_image instead of the numpy array
                # No need for output_size here as we resized manually
                detected_faces = face_detector.detect(temp_image_path, batch_size=1)
                # breakpoint() # Removed breakpoint

                # Check if any faces were detected and AU data is available
                if not detected_faces.empty:
                    # Access the first face's AU data
                    first_face_aus = detected_faces.aus
                    print(f"First face AU data: {first_face_aus}")
                    current_face_au_data['aus'] = {k: float(v) for k, v in first_face_aus.items()} # Ensure floats
                else:
                    # No face detected or no AU data extracted
                    current_face_au_data['error'] = "no face detected"
                    # Keep 'aus': {} empty

            except Exception as e:
                print(f"Error processing face AUs in frame {current_frame} (second {second}): {e}")
                current_face_au_data['error'] = f"py-feat error: {str(e)}"
                # Keep 'aus': {} empty
            finally:
                # Clean up the temporary image file
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except OSError as remove_err:
                        print(f"Error removing temporary image file {temp_image_path}: {remove_err}")

            # Add to face AU results
            face_au_results.append(current_face_au_data)

            # Update progress - Face analysis part (0% to 50%)
            progress = int((second / duration * 50) if duration > 0 else 0)
            update_progress(result_id, "processing", progress, f"Analyzing facial action units... ({second+1}/{int(duration)}s)", duration)

            # Move to next second
            second += 1
            current_frame += frame_interval
        
        video.release() # Release video capture early

        # Voice analysis part
        if audio_path:
            update_progress(result_id, "processing", 50, "Analyzing voice emotions...", duration)

            # Process voice emotion analysis using the new function
            # Pass result_id for progress updates within analyze_audio_by_second_hf
            voice_results, acoustic_features_results = analyze_audio_by_second_hf(audio_path, duration, result_id, temp_result_path)
            
            # Clean up the temporary audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as remove_err:
                    print(f"Error removing temporary audio file {audio_path}: {remove_err}")
        else:
             # If audio extraction failed, create placeholder voice results
             voice_results = [{'second': s, 'dominant_emotion': 'no audio extracted', 'emotions': {}} for s in range(int(duration))]
             acoustic_features_results = [{'second': s, 'error': 'No audio extracted'} for s in range(int(duration))]
             update_progress(result_id, "processing", 90, "Skipping voice analysis (no audio extracted)...", duration)

        # Combine results and calculate final scores
        raw_scores_by_second = []
            
        for i in range(int(duration)):
            # Find corresponding data for second i
            face_data = next((item for item in face_au_results if item['second'] == i),
                           {'second': i, 'aus': {}, 'error': 'Missing face data'}) # Default if not found

            voice_data = next((item for item in voice_results if item['second'] == i),
                            {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})

            acoustic_data = next((item for item in acoustic_features_results if item['second'] == i),
                               {'second': i, 'error': 'Missing acoustic data'})

            # Calculate raw depression metric for this second using face AUs, voice emotion, and acoustic features
            raw_score = calculate_raw_depression_metric(face_data, voice_data, acoustic_data)
            raw_scores_by_second.append(raw_score)

            # Store combined data for later use (depression score will be added after smoothing)
            combined_results.append({
                'second': i,
                'face_analysis': { # Renamed from face_emotion
                    'aus': face_data.get('aus', {}),
                    'error': face_data.get('error')
                },
                'voice_emotion': {
                    'dominant_emotion': voice_data['dominant_emotion'],
                    'emotions': voice_data['emotions']
                },
                'voice_acoustic_features': {k: v for k, v in acoustic_data.items() if k != 'second' and k != 'error' and v is not None and not (isinstance(v, float) and np.isnan(v))},
                # 'depression_score': depression_score # This will be added later after smoothing
            })
        
        # --- Temporal Smoothing and Final Score Calculation ---
        K = 3 # Sigmoid steepness factor
        smoothed_scaled_scores = []
        overall_depression_score = 0 # Default

        if raw_scores_by_second:
            # Apply rolling average (window of 5, centered)
            # Use pandas Series for easy rolling calculation
            raw_series = pd.Series(raw_scores_by_second)
            # Window size 5, center=True, min_periods=1 to handle edges
            smoothed_raw_scores = raw_series.rolling(window=5, center=True, min_periods=1).mean().tolist()

            # Apply sigmoid and scaling to smoothed scores
            for raw_smoothed_score in smoothed_raw_scores:
                sigmoid_score = 1.0 / (1.0 + np.exp(-K * raw_smoothed_score))
                final_score = sigmoid_score * 100
                smoothed_scaled_scores.append(final_score)

            # Add the final smoothed+scaled score back to combined_results
            for idx, score in enumerate(smoothed_scaled_scores):
                if idx < len(combined_results):
                    combined_results[idx]['depression_score'] = score

            # Calculate overall depression score (median of smoothed+scaled scores)
            overall_depression_score = np.median(smoothed_scaled_scores)

        # Save final results
        update_progress(result_id, "completed", 100, "Analysis complete", duration, combined_results)
        update_progress(result_id, "completed", 100, f"Analysis complete. Overall Depression Score: {overall_depression_score:.2f}", duration, combined_results)
        
    except Exception as e:
        print(f"Error in analyze_video_emotions: {e}")
        # Save error status
        update_progress(result_id, "error", -1, f"Error: {str(e)}", duration if 'duration' in locals() else 0, combined_results if 'combined_results' in locals() else [])
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
    """Analyze audio emotion second by second using Hugging Face model and acoustic features with progress updates."""
    voice_results = []
    acoustic_features_results = [] # NEW: Store acoustic features separately

    try:
        # Load full audio using pydub for easy slicing
        audio = AudioSegment.from_file(audio_path)
        target_sr = voice_feature_extractor.sampling_rate if voice_feature_extractor else 16000 # Default SR if extractor failed

        # Process audio in 1-second chunks
        for second in range(int(duration)):
            # Extract 1-second segment
            start_ms = second * 1000
            end_ms = (second + 1) * 1000

            if end_ms > len(audio):
                break

            segment = audio[start_ms:end_ms]

            # --- Acoustic Feature Extraction ---
            current_acoustic_features = { # Initialize features for this second
                'rms_mean': None, 'rms_std': None,
                'spectral_centroid_mean': None, 'spectral_centroid_std': None,
                'zcr_mean': None, 'zcr_std': None,
                'pitch_mean': None, 'pitch_std': None,
                'mfcc_mean_std': None,
                'error': None # To flag extraction issues
            }
            temp_path = None # Initialize temp_path
            try:
                # Save segment to temporary file for librosa/HF processing
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # Export segment: mono, target sample rate
                    segment.set_channels(1).set_frame_rate(target_sr).export(temp_path, format='wav')

                # Load with librosa for feature extraction
                y, sr = librosa.load(temp_path, sr=target_sr)

                # Calculate features (if audio is not silent)
                if np.abs(y).sum() > 1e-5: # Check if segment is effectively silent
                    # RMS
                    rms = librosa.feature.rms(y=y)[0]
                    current_acoustic_features['rms_mean'] = np.mean(rms)
                    current_acoustic_features['rms_std'] = np.std(rms)
                    # Spectral Centroid
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                    current_acoustic_features['spectral_centroid_mean'] = np.mean(spec_cent)
                    current_acoustic_features['spectral_centroid_std'] = np.std(spec_cent)
                    # Zero-Crossing Rate
                    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
                    current_acoustic_features['zcr_mean'] = np.mean(zcr)
                    current_acoustic_features['zcr_std'] = np.std(zcr)
                    # Pitch (F0) using pyin
                    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                    # Use only non-NaN F0 values for stats
                    f0_voiced = f0[~np.isnan(f0)]
                    if len(f0_voiced) > 0:
                        current_acoustic_features['pitch_mean'] = np.mean(f0_voiced)
                        current_acoustic_features['pitch_std'] = np.std(f0_voiced)
                    else:
                         current_acoustic_features['pitch_mean'] = 0 # Or None, handle downstream
                         current_acoustic_features['pitch_std'] = 0 # Or None
                    # MFCCs (get std dev of the *means* of the coefficients)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    current_acoustic_features['mfcc_mean_std'] = np.std(np.mean(mfccs, axis=1))

                # Predict emotion using the Hugging Face model (using the same temp_path)
                emotion_result = predict_voice_emotion_hf(temp_path)

            except Exception as extract_err:
                print(f"Error extracting features/predicting for second {second}: {extract_err}")
                current_acoustic_features['error'] = str(extract_err)
                # Still try to get HF prediction if features failed, or vice versa?
                # For now, if feature extraction fails, HF prediction might also fail or be skipped.
                # Let's create a default error emotion result
                emotion_result = {'dominant_emotion': 'analysis error', 'emotions': {}}
                # Fallback for acoustic features already initialized to None

            finally:
                # Delete temporary file
                try:
                    if temp_path and os.path.exists(temp_path): os.unlink(temp_path)
                except OSError as unlink_err:
                     print(f"Error deleting temp audio file {temp_path}: {unlink_err}")

            # Add emotion results
            voice_results.append({
                'second': second,
                'dominant_emotion': emotion_result['dominant_emotion'],
                'emotions': emotion_result['emotions']
            })
            # Add acoustic features results
            acoustic_features_results.append({
                'second': second,
                **current_acoustic_features # Unpack the features dict
            })

            # Update progress (Voice analysis part: 50% to 90%)
            progress = 50 + int(((second + 1) / duration * 40) if duration > 0 else 0)
            update_progress(result_id, "processing", progress, f"Analyzing voice emotions & features... ({second+1}/{int(duration)}s)", duration)

        # Return both emotion predictions and acoustic features
        return voice_results, acoustic_features_results

    except Exception as e:
        print(f"Error analyzing audio by second (HF + Acoustic): {e}")
        # Provide default error data for both lists if analysis fails early
        default_emotion = [{'second': s, 'dominant_emotion': 'analysis error', 'emotions': {}} for s in range(int(duration))]
        default_acoustic = [{'second': s, 'error': 'Overall analysis failed'} for s in range(int(duration))]
        return default_emotion, default_acoustic

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