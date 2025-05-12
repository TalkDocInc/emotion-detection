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
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoModel, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
import whisper
import pandas as pd

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

# Voice Emotion Model
VOICE_MODEL_NAME = "superb/wav2vec2-base-superb-er"
try:
    voice_config = AutoConfig.from_pretrained(VOICE_MODEL_NAME)
    voice_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(VOICE_MODEL_NAME)
    voice_model = AutoModel.from_pretrained(VOICE_MODEL_NAME)
    emotion_labels = ['neu', 'hap', 'ang', 'sad']
    voice_config.id2label = {i: label for i, label in enumerate(emotion_labels)}
    voice_config.label2id = {label: i for i, label in enumerate(emotion_labels)}
    print(f"Successfully loaded voice model: {VOICE_MODEL_NAME}")
except Exception as e:
    print(f"Error loading Hugging Face voice model {VOICE_MODEL_NAME}: {e}")
    voice_feature_extractor, voice_model, voice_config = None, None, None

# Speech Transcription Model (Whisper)
try:
    # Using the base model for balance between speed and accuracy
    whisper_model = whisper.load_model("base")
    print("Successfully loaded Whisper model (base)")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

# Text Depression Analysis Model (DepRoBERTa)
TEXT_DEPRESSION_MODEL_NAME = "rafalposwiata/deproberta-large-depression"
try:
    text_depression_tokenizer = AutoTokenizer.from_pretrained(TEXT_DEPRESSION_MODEL_NAME)
    text_depression_model = AutoModelForSequenceClassification.from_pretrained(TEXT_DEPRESSION_MODEL_NAME)
    # Create a pipeline for easier inference
    text_depression_pipeline = hf_pipeline(
        "text-classification",
        model=text_depression_model,
        tokenizer=text_depression_tokenizer,
        return_all_scores=True # Get scores for all labels
    )
    print(f"Successfully loaded text depression model: {TEXT_DEPRESSION_MODEL_NAME}")
except Exception as e:
    print(f"Error loading Hugging Face text depression model {TEXT_DEPRESSION_MODEL_NAME}: {e}")
    text_depression_tokenizer, text_depression_model, text_depression_pipeline = None, None, None

# --- End Hugging Face Model Loading ---

# --- Depression Analysis Configuration ---

# Numerical mapping for text depression labels
TEXT_DEPRESSION_LABEL_TO_SCORE = {
    'not depression': 0.0, # Assuming lower is better
    'moderate': 0.5,
    'severe': 1.0
}
# Fallback score if text analysis fails
DEFAULT_TEXT_DEPRESSION_SCORE = 0.0

# Depression indicators in facial emotions (raw score contribution, approx -1 to 1)
FACE_DEPRESSION_WEIGHTS = {
    'angry': 0.6, 'disgust': 0.5, 'fear': 0.7, 'happy': -0.8,
    'sad': 0.9, 'surprise': 0.0, 'neutral': 0.05,
    'no face detected': 0.0
}

# Depression indicators in voice emotions (raw score contribution, approx -1 to 1)
VOICE_DEPRESSION_WEIGHTS = {
    'neu': 0.05, 'hap': -0.8, 'ang': 0.5, 'sad': 0.9,
    'no audio extracted': 0.0, 'audio export error': 0.0, 'model error': 0.0,
    'prediction error': 0.0, 'analysis error': 0.0, 'no data': 0.0
}

# Parameters for scaling acoustic features
ACOUSTIC_FEATURE_SCALING_PARAMS = {
    'rms_mean': {'center': 0.05, 'scale': 0.05}, 'rms_std': {'center': 0.01, 'scale': 0.01},
    'spectral_centroid_mean': {'center': 1500, 'scale': 1000}, 'spectral_centroid_std': {'center': 500, 'scale': 500},
    'zcr_mean': {'center': 0.1, 'scale': 0.1}, 'zcr_std': {'center': 0.05, 'scale': 0.05},
    'pitch_mean': {'center': 150, 'scale': 50}, 'pitch_std': {'center': 20, 'scale': 20},
    'mfcc_mean_std': {'center': 10, 'scale': 10}
}

# Weights for Granular Acoustic Features (applied AFTER scaling to approx [-1, 1])
VOICE_ACOUSTIC_DEPRESSION_WEIGHTS = {
    'rms_mean': -0.3, 'rms_std': -0.4,
    'spectral_centroid_mean': -0.2, 'spectral_centroid_std': -0.2,
    'zcr_mean': 0.0, 'zcr_std': 0.0,
    'pitch_mean': -0.5, 'pitch_std': -0.7,
    'mfcc_mean_std': -0.3
}

# Weighting *within* the voice component
VOICE_EMOTION_WEIGHT = 0.0 # Contribution from categorical emotion (e.g., 'sad')
VOICE_ACOUSTIC_WEIGHT = 1.0 # Contribution from granular features (pitch, energy, etc.)

# Overall weighting between face, voice (combined), and text analysis for FINAL depression score
# Adjusted to include text analysis
FACE_OVERALL_WEIGHT = 0.0 # Face contributes 30%
VOICE_OVERALL_WEIGHT = 0.7 # Voice (emotion + acoustics) contributes 30%
TEXT_OVERALL_WEIGHT = 0.3 # Text analysis contributes 40

# --- End Depression Analysis Configuration ---

# Preload/reuse DeepFace emotion model
try:
    deepface_emotion_model = DeepFace.build_model('VGG-Face')
    print("Successfully preloaded DeepFace model (VGG-Face)")
except Exception as e:
    print(f"Error preloading DeepFace model: {e}")
    deepface_emotion_model = None # Handle potential loading error

class EmotionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        # Add handling for Path objects if whisper returns them
        if isinstance(obj, pathlib.Path):
             return str(obj)
        return super().default(obj)

# Calculates the raw weighted score PER SECOND based on face/voice/acoustics
# This score is calculated BEFORE combining with the overall text score and final scaling
def calculate_per_second_raw_depression_metric(face_emotion_data, voice_emotion_data, voice_acoustic_features):
    """
    Calculate a raw depression metric for a single second based on face, voice emotion,
    and voice acoustic features. Uses detailed emotion scores when available.
    The score is roughly in [-1, 1].

    Args:
        face_emotion_data: Dict possibly containing 'dominant_emotion' and 'emotions'.
        voice_emotion_data: Dict possibly containing 'dominant_emotion' and 'emotions'.
        voice_acoustic_features: Dict containing extracted acoustic features for the second.

    Returns:
        A raw score roughly between -1 and 1, representing weighted depression indication
        based on face/voice/acoustics for this second.
    """
    face_dep_contribution = 0.0
    voice_dep_contribution = 0.0 # Combined voice score for the second
    face_data_available = False
    voice_data_available = False # Flag if any voice data (emotion or acoustic) is useful

    # --- Calculate Face Depression Contribution ---
    face_emotions = face_emotion_data.get('emotions', {})
    face_dominant = face_emotion_data.get('dominant_emotion') # Can be None or 'no face detected'

    if face_emotions:
        total_face_score = sum(face_emotions.values())
        if total_face_score > 1e-6:
            for emotion, score in face_emotions.items():
                if emotion in FACE_DEPRESSION_WEIGHTS:
                    face_dep_contribution += (score / total_face_score) * FACE_DEPRESSION_WEIGHTS[emotion]
            face_data_available = True
        # Fallback to dominant if scores are zero but dominant exists and has weight
        elif face_dominant and face_dominant in FACE_DEPRESSION_WEIGHTS and abs(FACE_DEPRESSION_WEIGHTS[face_dominant]) > 1e-6:
             face_dep_contribution = FACE_DEPRESSION_WEIGHTS[face_dominant]
             face_data_available = True
    # Use dominant if no detailed scores but dominant exists and has weight
    elif face_dominant and face_dominant in FACE_DEPRESSION_WEIGHTS and abs(FACE_DEPRESSION_WEIGHTS[face_dominant]) > 1e-6:
        face_dep_contribution = FACE_DEPRESSION_WEIGHTS[face_dominant]
        face_data_available = True

    # --- Calculate Voice Depression Contribution (Combined Emotion + Acoustics) ---
    voice_emotion_contribution = 0.0
    voice_acoustic_contribution = 0.0
    emotion_data_used = False
    acoustic_data_used = False

    # 1. Contribution from Voice Emotion Category
    voice_emotions = voice_emotion_data.get('emotions', {})
    voice_dominant = voice_emotion_data.get('dominant_emotion') # Can be None or error states

    if voice_emotions:
        total_voice_score = sum(voice_emotions.values())
        if total_voice_score > 1e-6:
            for emotion, score in voice_emotions.items():
                if emotion in VOICE_DEPRESSION_WEIGHTS:
                    voice_emotion_contribution += (score / total_voice_score) * VOICE_DEPRESSION_WEIGHTS[emotion]
            emotion_data_used = True
        elif voice_dominant and voice_dominant in VOICE_DEPRESSION_WEIGHTS and abs(VOICE_DEPRESSION_WEIGHTS[voice_dominant]) > 1e-6:
            voice_emotion_contribution = VOICE_DEPRESSION_WEIGHTS[voice_dominant]
            emotion_data_used = True
    elif voice_dominant and voice_dominant in VOICE_DEPRESSION_WEIGHTS and abs(VOICE_DEPRESSION_WEIGHTS[voice_dominant]) > 1e-6:
        voice_emotion_contribution = VOICE_DEPRESSION_WEIGHTS[voice_dominant]
        emotion_data_used = True

    # 2. Contribution from Acoustic Features
    if voice_acoustic_features and voice_acoustic_features.get('error') is None:
        num_acoustic_features_used = 0
        temp_acoustic_score = 0.0
        for feature_name, weight in VOICE_ACOUSTIC_DEPRESSION_WEIGHTS.items():
            feature_value = voice_acoustic_features.get(feature_name)
            scaling_params = ACOUSTIC_FEATURE_SCALING_PARAMS.get(feature_name)

            if (feature_value is not None and not np.isnan(feature_value) and
                scaling_params and scaling_params.get('scale') is not None and
                abs(scaling_params['scale']) > 1e-9 and abs(weight) > 1e-6):

                center = scaling_params.get('center', 0.0)
                scale = scaling_params['scale']
                scaled_value = np.tanh((feature_value - center) / scale) # Scale to approx [-1, 1]
                temp_acoustic_score += scaled_value * weight
                num_acoustic_features_used += 1

        if num_acoustic_features_used > 0:
            # Use the direct weighted sum of scaled features
            voice_acoustic_contribution = temp_acoustic_score
            acoustic_data_used = True

    # 3. Combine Voice Contributions (Emotion + Acoustic) for the second
    if emotion_data_used and acoustic_data_used:
        voice_dep_contribution = (voice_emotion_contribution * VOICE_EMOTION_WEIGHT) + \
                                 (voice_acoustic_contribution * VOICE_ACOUSTIC_WEIGHT)
        voice_data_available = True
    elif emotion_data_used:
        voice_dep_contribution = voice_emotion_contribution
        voice_data_available = True
    elif acoustic_data_used:
        voice_dep_contribution = voice_acoustic_contribution
        voice_data_available = True
    # Else: voice_dep_contribution remains 0.0, voice_data_available remains False

    # --- Combine Face and Voice scores for this second ---
    # Note: We do NOT use FACE_OVERALL_WEIGHT or VOICE_OVERALL_WEIGHT here.
    # Those are applied *after* averaging/medianing these per-second scores.
    # Here, we just calculate a combined face/voice score for the second.
    # If only one modality is available, we use its score directly.
    # If both are available, we could average them or use pre-defined weights *specific to this stage*
    # For simplicity, let's average if both available, otherwise take the available one.

    if face_data_available and voice_data_available:
        # Simple average for now, could be weighted differently if needed
        combined_per_second_score = (face_dep_contribution + voice_dep_contribution) / 2.0
    elif face_data_available:
        combined_per_second_score = face_dep_contribution
    elif voice_data_available:
        combined_per_second_score = voice_dep_contribution
    else:
        combined_per_second_score = 0.0 # No data for this second

    return combined_per_second_score

# --- NEW: Functions for Transcription and Text Analysis ---

def transcribe_audio(audio_path):
    """Transcribes the given audio file using the loaded Whisper model."""
    if whisper_model is None:
        print("Whisper model not loaded. Skipping transcription.")
        return None, "Whisper model not loaded"

    try:
        # Perform transcription
        # Consider adding options like language detection if needed
        result = whisper_model.transcribe(audio_path)
        return result["text"], None # Return transcript text
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return None, str(e)

def analyze_text_depression(text):
    """Analyzes the transcribed text for depression using the loaded DepRoBERTa pipeline."""
    if text_depression_pipeline is None:
        print("Text depression model pipeline not loaded. Skipping analysis.")
        return {"label": "model error", "score": DEFAULT_TEXT_DEPRESSION_SCORE, "error": "Model not loaded"}

    if not text or not isinstance(text, str):
         print("Invalid text input for depression analysis.")
         return {"label": "input error", "score": DEFAULT_TEXT_DEPRESSION_SCORE, "error": "Invalid text"}

    try:
        # The pipeline returns a list of lists (one for each input string)
        # Each inner list contains dicts for each label and its score
        results = text_depression_pipeline(text)[0] # Get results for the first (only) input string
        print("Results: ", results)

        # Find the label with the highest score (still needed for "dominant_label" in output)
        best_result = max(results, key=lambda x: x['score'])
        print("Best result: ", best_result)
        dominant_label = best_result['label']
        print("Dominant label: ", dominant_label)

        # Calculate numerical_score as a weighted average of all label scores
        # based on their model confidence and predefined severity.
        numerical_score = 0.0
        if results: # Ensure there are results to process
            for res_item in results:
                label_name = res_item['label']
                model_confidence = res_item['score']
                # Get the predefined severity for this label, default to 0.0 if not found
                label_severity_score = TEXT_DEPRESSION_LABEL_TO_SCORE.get(label_name, 0.0)
                numerical_score += model_confidence * label_severity_score
                print("Numerical score: ", numerical_score)
        else: # Fallback if results are empty for some reason
            numerical_score = DEFAULT_TEXT_DEPRESSION_SCORE

        print("Numerical score: ", numerical_score) # This will now reflect the new calculation

        # Return label, numerical score, and full results
        return {
            "dominant_label": dominant_label,
            "numerical_score": numerical_score,
            "all_scores": {res['label']: res['score'] for res in results}, # Store all label scores
            "error": None
        }

    except Exception as e:
        print(f"Error during text depression analysis: {e}")
        return {"label": "analysis error", "score": DEFAULT_TEXT_DEPRESSION_SCORE, "error": str(e)}

# --- End NEW Functions ---

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
        # Create a unique directory for this analysis
        result_id = str(uuid.uuid4())
        analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], result_id)
        os.makedirs(analysis_dir, exist_ok=True)

        unique_filename = f"{uuid.uuid4()}_{filename}" # Keep original filename logic if needed elsewhere
        file_path = os.path.join(analysis_dir, unique_filename)
        file.save(file_path)

        # Get the bypass_sigmoid option
        bypass_sigmoid = 'bypass_sigmoid' in request.form

        # Start analysis in a background thread
        thread = threading.Thread(target=analyze_video_emotions, args=(file_path, result_id, analysis_dir, bypass_sigmoid))
        thread.daemon = True
        thread.start()

        return jsonify({
            'message': 'Video upload processing started',
            'filename': unique_filename, # Or maybe just result_id?
            'result_id': result_id
        })

    else:
        return jsonify({'error': 'File type not allowed'}), 400

def analyze_video_emotions(video_path, result_id, analysis_dir, bypass_sigmoid=False):
    """Analyze emotions in the video using face, voice, and text analysis"""
    face_results_per_second = []
    voice_results_per_second = []
    acoustic_features_per_second = []
    combined_results_per_second = [] # Store detailed per-second data
    raw_scores_per_second = [] # Store the combined face/voice/acoustic raw score for each second
    final_results = { # Structure for the final JSON output
        "status": "processing", "progress": 0, "message": "Initializing...",
        "total_seconds": 0, "overall_depression_score": 0,
        "transcript": None, "transcript_error": None,
        "text_depression_analysis": None,
        "results": [] # Per-second detailed results
    }
    temp_result_path = os.path.join(analysis_dir, f"{result_id}_progress.json")
    audio_path = None

    def update_progress(progress, message):
        final_results["progress"] = progress
        final_results["message"] = message
        try:
            with open(temp_result_path, 'w') as f:
                json.dump(final_results, f, cls=EmotionEncoder, indent=2)
        except Exception as write_err:
            print(f"Error writing progress file: {write_err}")

    try:
        # Check if models are loaded
        if deepface_emotion_model is None:
            raise ValueError("DeepFace model failed to load.")
        if voice_feature_extractor is None or voice_model is None:
            raise ValueError("Voice emotion model failed to load.")
        # Don't fail immediately if whisper/text models failed, allow partial analysis

        update_progress(1, "Opening video file...")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
             raise IOError(f"Cannot open video file: {video_path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        final_results["total_seconds"] = int(duration)
        if duration == 0:
             raise ValueError("Video duration is zero or could not be determined.")

        update_progress(5, "Extracting audio...")
        # Use analysis_dir for audio output
        audio_path = extract_audio_from_video(video_path, output_dir=analysis_dir)
        if not audio_path:
            final_results["message"] = "Audio extraction failed. Skipping voice and text analysis."
            # Continue with face analysis only

        # --- NEW: Transcription ---
        transcript = None
        transcript_error = None
        if audio_path and whisper_model:
            update_progress(10, "Transcribing audio (may take time)...")
            transcript, transcript_error = transcribe_audio(audio_path)
            final_results["transcript"] = transcript
            final_results["transcript_error"] = transcript_error
            if transcript_error:
                 final_results["message"] = f"Transcription failed: {transcript_error}. Proceeding without text analysis."
                 update_progress(15, final_results["message"])
            else:
                 update_progress(15, "Transcription complete.")
        elif not audio_path:
            final_results["transcript_error"] = "No audio extracted"
            update_progress(15, "Skipping transcription (no audio).")
        else: # Whisper model failed to load
            final_results["transcript_error"] = "Whisper model not loaded"
            update_progress(15, "Skipping transcription (model load failed).")

        # --- NEW: Text Depression Analysis ---
        text_analysis_result = None
        if transcript and text_depression_pipeline:
            update_progress(20, "Analyzing text for depression...")
            text_analysis_result = analyze_text_depression(transcript)
            final_results["text_depression_analysis"] = text_analysis_result
            update_progress(25, "Text analysis complete.")
        elif not transcript:
            final_results["text_depression_analysis"] = {"error": "No transcript available"}
            update_progress(25, "Skipping text analysis (no transcript).")
        else: # Text model failed to load
            final_results["text_depression_analysis"] = {"error": "Text depression model not loaded"}
            update_progress(25, "Skipping text analysis (model load failed).")

        # --- Face Analysis Loop ---
        update_progress(30, "Analyzing facial emotions...")
        frame_interval = int(fps) if fps > 0 else 1
        current_frame = 0
        second = 0
        while current_frame < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = video.read()
            if not ret: break

            try:
                # Use the preloaded model
                emotion_analysis = DeepFace.analyze(frame, actions=['emotion'],
                                                    enforce_detection=False, silent=True,
                                                    detector_backend='opencv') # Specify backend for consistency
                data = emotion_analysis[0] if isinstance(emotion_analysis, list) else emotion_analysis
                face_results_per_second.append({
                    'second': second, 'dominant_emotion': data['dominant_emotion'], 'emotions': data['emotion']
                })
            except ValueError as ve: # Handles "Face could not be detected"
                 face_results_per_second.append({'second': second, 'dominant_emotion': 'no face detected', 'emotions': {}})
            except Exception as e:
                print(f"Error processing face in frame {current_frame}: {e}")
                face_results_per_second.append({'second': second, 'dominant_emotion': 'analysis error', 'emotions': {}})

            progress = 30 + int((second / duration * 25) if duration > 0 else 0) # Face: 30% -> 55%
            update_progress(progress, f"Analyzing facial emotions... ({second+1}/{int(duration)}s)")
            second += 1
            current_frame += frame_interval
        video.release()

        # --- Voice Analysis (Emotion + Acoustic Features) ---
        if audio_path and voice_model: # Only run if audio exists and model loaded
            update_progress(55, "Analyzing voice emotions and features...")
            # Analyze audio second by second (returns voice emotion and acoustic features)
            voice_results_per_second, acoustic_features_per_second = analyze_audio_by_second_hf(
                audio_path, duration, result_id, temp_result_path, start_progress=55, progress_range=35 # Voice: 55% -> 90%
            )
        elif not audio_path:
             update_progress(90, "Skipping voice analysis (no audio extracted)...")
             # Create placeholders if needed downstream
             voice_results_per_second = [{'second': s, 'dominant_emotion': 'no audio extracted', 'emotions': {}} for s in range(int(duration))]
             acoustic_features_per_second = [{'second': s, 'error': 'No audio extracted'} for s in range(int(duration))]
        else: # Voice model failed
             update_progress(90, "Skipping voice analysis (model load failed)...")
             voice_results_per_second = [{'second': s, 'dominant_emotion': 'model error', 'emotions': {}} for s in range(int(duration))]
             acoustic_features_per_second = [{'second': s, 'error': 'Model load failed'} for s in range(int(duration))]

        # --- Combine Per-Second Results and Calculate Raw Scores ---
        update_progress(90, "Combining per-second results...")
        for i in range(int(duration)):
            face_data = next((item for item in face_results_per_second if item['second'] == i),
                           {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})
            voice_data = next((item for item in voice_results_per_second if item['second'] == i),
                            {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})
            acoustic_data = next((item for item in acoustic_features_per_second if item['second'] == i),
                               {'second': i, 'error': 'Missing data'})

            # Calculate raw score for this second based on face/voice/acoustics
            raw_score = calculate_per_second_raw_depression_metric(face_data, voice_data, acoustic_data)
            raw_scores_per_second.append(raw_score)

            # Store detailed combined data for this second
            combined_results_per_second.append({
                'second': i,
                'face_emotion': face_data,
                'voice_emotion': voice_data,
                'voice_acoustic_features': {k: v for k, v in acoustic_data.items() if k != 'second' and k != 'error'},
                'raw_depression_score_fv': raw_score # Store the face/voice/acoustic raw score
                # Final scaled score will be added later
            })
        final_results["results"] = combined_results_per_second # Add per-second details to final output

        # --- Calculate Overall Face/Voice/Acoustic Score ---
        update_progress(95, "Calculating overall scores...")
        median_raw_fv_score = 0.0
        if raw_scores_per_second:
            # Apply temporal smoothing (rolling average) to raw face/voice scores
            raw_series = pd.Series(raw_scores_per_second)
            smoothed_raw_fv_scores = raw_series.rolling(window=5, center=True, min_periods=1).mean().tolist()
            # Use the median of the *smoothed* raw scores as the representative FV score
            median_raw_fv_score = np.median(smoothed_raw_fv_scores) if smoothed_raw_fv_scores else 0.0

        # --- Combine with Text Score ---
        # Get the numerical score from text analysis, default if failed
        text_dep_score = text_analysis_result.get("numerical_score", DEFAULT_TEXT_DEPRESSION_SCORE) if text_analysis_result else DEFAULT_TEXT_DEPRESSION_SCORE
        # Determine if text score is valid/usable (i.e., analysis didn't error out)
        text_score_available = text_analysis_result and text_analysis_result.get("error") is None

        # Check if face/voice score is valid (i.e., we got some data)
        fv_score_available = bool(raw_scores_per_second) # True if the list is not empty

        # Calculate the final combined raw score using overall weights
        # Adjust weights based on availability
        final_raw_score = 0.0
        total_weight_used = 0.0
        if fv_score_available:
            total_weight_used += FACE_OVERALL_WEIGHT + VOICE_OVERALL_WEIGHT # Treat F/V block together
        if text_score_available:
            total_weight_used += TEXT_OVERALL_WEIGHT

        if total_weight_used > 1e-6:
            # Normalize weights
            adjusted_fv_weight = (FACE_OVERALL_WEIGHT + VOICE_OVERALL_WEIGHT) / total_weight_used if fv_score_available else 0
            adjusted_text_weight = TEXT_OVERALL_WEIGHT / total_weight_used if text_score_available else 0

            final_raw_score = (median_raw_fv_score * adjusted_fv_weight) + \
                              (text_dep_score * adjusted_text_weight) # Using the 0-1 text score directly here
        else:
            final_raw_score = 0.0 # No data from any source

        # --- Final Scaling and Smoothing (Apply sigmoid to smoothed FV scores for per-second timeline) ---
        K = 3 # Sigmoid steepness factor
        final_scaled_scores_per_second = []
        overall_final_score = 0 # Default final score (0-100)

        if smoothed_raw_fv_scores: # Check if we have smoothed scores to work with
            for raw_smoothed_score in smoothed_raw_fv_scores:
                # Apply sigmoid and scale to 0-100 for the timeline display
                # This timeline score calculation is independent of the overall score bypass
                sigmoid_score = 1.0 / (1.0 + np.exp(-K * raw_smoothed_score))
                final_scaled_score = sigmoid_score * 100
                final_scaled_scores_per_second.append(final_scaled_score)

            # Add the final scaled score back to per-second results
            for idx, score in enumerate(final_scaled_scores_per_second):
                if idx < len(final_results["results"]):
                    final_results["results"][idx]['final_depression_score'] = score

            # Calculate the OVERALL final score (0-100)
            if bypass_sigmoid:
                # Ensure final_raw_score is clamped to a reasonable range (e.g., 0 to 1, or -1 to 1 if that's its natural range)
                # Assuming final_raw_score from text analysis is already 0-1, or combined fv is also scaled to a similar range
                # If final_raw_score can be outside 0-1, this direct scaling might need adjustment
                # For now, assuming final_raw_score is intended to be in a range that makes sense to scale by 100.
                # If text_dep_score (0-1) is the only contributor to final_raw_score, this will be text_dep_score * 100
                print("Text depression score: ", text_dep_score)
                print("Final raw score: ", final_raw_score)
                print("Overall final score: ", overall_final_score)
                overall_final_score = np.clip(final_raw_score * 100, 0, 100)
            else:
                # Apply sigmoid to the weighted average of median_fv and text_score
                final_sigmoid = 1.0 / (1.0 + np.exp(-K * final_raw_score))
                overall_final_score = final_sigmoid * 100

        final_results["overall_depression_score"] = overall_final_score
        final_results["status"] = "completed"
        final_results["progress"] = 100
        final_results["message"] = "Analysis complete."

        # Save final combined results
        with open(temp_result_path, 'w') as f:
             json.dump(final_results, f, cls=EmotionEncoder, indent=2)

    except Exception as e:
        print(f"Error in analyze_video_emotions (result_id: {result_id}): {e}")
        final_results["status"] = "error"
        final_results["error"] = str(e)
        final_results["progress"] = 100 # Mark as finished even on error
        # Save error status
        try:
            with open(temp_result_path, 'w') as f:
                json.dump(final_results, f, cls=EmotionEncoder, indent=2)
        except Exception as write_err:
             print(f"Critical error: Failed to write error status file: {write_err}")

    finally:
        # Ensure video capture is released
        if 'video' in locals() and video.isOpened():
            video.release()
        # Clean up temporary audio path ONLY if it wasn't placed in the analysis_dir
        # If extract_audio_from_video places it outside analysis_dir, clean it up.
        # Current implementation puts it inside analysis_dir, so we leave it.
        # if audio_path and os.path.exists(audio_path) and os.path.dirname(audio_path) != analysis_dir:
        #      try:
        #          os.remove(audio_path)
        #      except OSError:
        #          pass
        # Optional: Clean up original uploaded video? Or leave it in analysis_dir?
        # if os.path.exists(video_path):
        #     try:
        #         os.remove(video_path)
        #     except OSError:
        #         pass
        print(f"Analysis thread finished for {result_id}. Status: {final_results['status']}")

# Modified to handle progress updates within the voice analysis stage
def analyze_audio_by_second_hf(audio_path, duration, result_id, temp_result_path, start_progress=50, progress_range=40):
    """Analyze audio emotion second by second using Hugging Face model and acoustic features with progress updates."""
    voice_results = []
    acoustic_features_results = []
    current_progress_data = {} # To load/save progress

    # Function to update progress within this stage
    def update_audio_progress(second):
        progress = start_progress + int(((second + 1) / duration * progress_range) if duration > 0 else 0)
        message = f"Analyzing voice emotions & features... ({second+1}/{int(duration)}s)"
        # Load existing data to preserve other fields (like transcript)
        try:
            with open(temp_result_path, 'r') as f:
                 current_progress_data = json.load(f)
        except Exception:
             current_progress_data = {"status": "processing", "total_seconds": int(duration), "results": []} # Basic fallback

        current_progress_data["progress"] = progress
        current_progress_data["message"] = message
        try:
            with open(temp_result_path, 'w') as f:
                 json.dump(current_progress_data, f, cls=EmotionEncoder, indent=2)
        except Exception as write_err:
            print(f"Error writing audio progress: {write_err}")

    try:
        audio = AudioSegment.from_file(audio_path)
        target_sr = voice_feature_extractor.sampling_rate if voice_feature_extractor else 16000

        for second in range(int(duration)):
            start_ms = second * 1000
            end_ms = (second + 1) * 1000
            if end_ms > len(audio): break
            segment = audio[start_ms:end_ms]

            current_acoustic_features = {
                'rms_mean': None, 'rms_std': None, 'spectral_centroid_mean': None,
                'spectral_centroid_std': None, 'zcr_mean': None, 'zcr_std': None,
                'pitch_mean': None, 'pitch_std': None, 'mfcc_mean_std': None, 'error': None
            }
            emotion_result = {'dominant_emotion': 'pending', 'emotions': {}} # Default before processing
            temp_path = None

            try:
                # Need a temp file for librosa and HF model
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    segment.set_channels(1).set_frame_rate(target_sr).export(temp_path, format='wav')

                y, sr = librosa.load(temp_path, sr=target_sr)

                # Acoustic Feature Extraction
                if np.abs(y).sum() > 1e-5:
                    try: rms = librosa.feature.rms(y=y)[0]; current_acoustic_features['rms_mean']=np.mean(rms); current_acoustic_features['rms_std']=np.std(rms)
                    except Exception as fe: print(f"RMS Err {second}: {fe}")
                    try: spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]; current_acoustic_features['spectral_centroid_mean']=np.mean(spec_cent); current_acoustic_features['spectral_centroid_std']=np.std(spec_cent)
                    except Exception as fe: print(f"SpecCent Err {second}: {fe}")
                    try: zcr = librosa.feature.zero_crossing_rate(y=y)[0]; current_acoustic_features['zcr_mean']=np.mean(zcr); current_acoustic_features['zcr_std']=np.std(zcr)
                    except Exception as fe: print(f"ZCR Err {second}: {fe}")
                    try:
                        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                        f0_voiced = f0[~np.isnan(f0)]
                        if len(f0_voiced) > 0: current_acoustic_features['pitch_mean']=np.mean(f0_voiced); current_acoustic_features['pitch_std']=np.std(f0_voiced)
                        else: current_acoustic_features['pitch_mean']=0; current_acoustic_features['pitch_std']=0
                    except Exception as fe: print(f"Pitch Err {second}: {fe}"); current_acoustic_features['pitch_mean']=0; current_acoustic_features['pitch_std']=0
                    try: mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13); current_acoustic_features['mfcc_mean_std']=np.std(np.mean(mfccs, axis=1))
                    except Exception as fe: print(f"MFCC Err {second}: {fe}")

                # Voice Emotion Prediction (using the same temp file)
                emotion_result = predict_voice_emotion_hf(temp_path)

            except Exception as extract_err:
                print(f"Error processing audio segment {second}: {extract_err}")
                current_acoustic_features['error'] = str(extract_err)
                emotion_result = {'dominant_emotion': 'analysis error', 'emotions': {}}

            finally:
                if temp_path and os.path.exists(temp_path):
                    try: os.unlink(temp_path)
                    except OSError as unlink_err: print(f"Error deleting temp audio file {temp_path}: {unlink_err}")

            # Append results for this second
            voice_results.append({'second': second, **emotion_result})
            acoustic_features_results.append({'second': second, **current_acoustic_features})

            # Update overall progress file
            update_audio_progress(second)

        return voice_results, acoustic_features_results

    except Exception as e:
        print(f"Major error in analyze_audio_by_second_hf: {e}")
        # Provide default error data
        duration_int = int(duration) if duration else 0
        default_emotion = [{"second": s, "dominant_emotion": "analysis error", "emotions": {}} for s in range(duration_int)]
        default_acoustic = [{"second": s, "error": "Overall audio analysis failed"} for s in range(duration_int)]
        return default_emotion, default_acoustic

@app.route('/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Get the current results/progress for a video analysis"""
    # Look for the progress file within the analysis directory
    analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], result_id)
    result_path = os.path.join(analysis_dir, f"{result_id}_progress.json")

    if not os.path.exists(result_path):
        # Check if the directory itself exists - indicates maybe upload failed early
        if not os.path.exists(analysis_dir):
             return jsonify({"status": "not_found", "message": "Analysis ID not found."}), 404
        else:
             # Directory exists, but no progress file yet - implies it's starting
             return jsonify({"status": "initializing", "message": "Analysis initializing..."}), 202 # Accepted

    try:
        with open(result_path, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    except json.JSONDecodeError:
         return jsonify({"status": "error", "message": "Failed to read progress file."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500

@app.route('/uploads/<result_id>/<filename>')
def uploaded_file(result_id, filename):
    """Serve files from a specific analysis directory"""
    analysis_dir = os.path.join(app.config['UPLOAD_FOLDER'], result_id)
    # Basic security check: ensure filename doesn't try to escape the directory
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    return send_from_directory(analysis_dir, filename)

def extract_audio_from_video(video_path, output_dir=None):
    """Extract audio from video file into the specified output directory"""
    if not output_dir:
        output_dir = os.path.dirname(video_path) # Default to same dir if not specified

    try:
        base_name = os.path.basename(video_path)
        # Use a consistent naming scheme within the output_dir
        audio_name = f"{os.path.splitext(base_name)[0]}_audio.wav"
        audio_path = os.path.join(output_dir, audio_name)

        video = VideoFileClip(video_path)
        # Ensure audio codec is suitable for Whisper/Librosa (e.g., PCM WAV)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        video.close() # Close the video file handle
        return audio_path
    except AttributeError:
         print(f"Video file {video_path} might not contain an audio stream.")
         return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        if 'video' in locals() and video: video.close() # Ensure closure on error
        return None

def predict_voice_emotion_hf(audio_segment_path):
    """Predict emotion from a 1-second audio segment using Hugging Face wav2vec2 model."""
    if voice_feature_extractor is None or voice_model is None or voice_config is None:
        return {'dominant_emotion': 'model error', 'emotions': {}}

    try:
        target_sr = voice_feature_extractor.sampling_rate
        speech, sr = librosa.load(audio_segment_path, sr=target_sr)

        # Handle potentially short/empty segments after loading
        if len(speech) == 0:
             return {'dominant_emotion': 'empty segment', 'emotions': {}}

        inputs = voice_feature_extractor(speech, sampling_rate=target_sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = voice_model(**inputs)
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)

            # Simple linear layer simulation for IEMOCAP 4-class (adjust if model has classifier head)
            # This part is heuristic if the base wav2vec model doesn't have an ER head
            num_labels = len(voice_config.id2label)
            logits = pooled_output[:, :num_labels] # Assume first N outputs correspond to labels
            scores_tensor = torch.softmax(logits, dim=1)
            scores = scores_tensor.squeeze().tolist()
            predicted_class_id = torch.argmax(scores_tensor, dim=1).item()

            # Ensure scores list matches number of labels
            if len(scores) < num_labels:
                 scores.extend([0.0] * (num_labels - len(scores))) # Pad with zeros if needed

            dominant_emotion = voice_config.id2label.get(predicted_class_id, 'unknown')

        emotion_scores_dict = {voice_config.id2label[i]: scores[i] for i in range(num_labels)}

        return {'dominant_emotion': dominant_emotion, 'emotions': emotion_scores_dict}

    except Exception as e:
        print(f"Error predicting voice emotion with HF model: {e}")
        return {'dominant_emotion': 'prediction error', 'emotions': {}}

if __name__ == '__main__':
    print("Starting Flask app...")
    # Consider security implications of running in debug mode in production
    app.run(debug=True, host='0.0.0.0', port=5001) # Example: Use a different port
    print("Flask App finished.")