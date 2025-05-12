import os
import cv2
import numpy as np
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
import time
import json
import threading
import shutil
from datetime import datetime, timedelta
import librosa
import soundfile as sf
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
import tempfile
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, AutoModel, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
import whisper
import pandas as pd
from pyfeat import Detector
import pathlib

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
# This is the old map based on categorical emotions, will be less used for py-feat.
FACE_DEPRESSION_WEIGHTS = {
    'angry': 0.6, 'disgust': 0.5, 'fear': 0.7, 'happy': -0.8,
    'sad': 0.9, 'surprise': 0.0, 'neutral': 0.05,
    'no face detected': 0.0  # py-feat specific error
}

# --- NEW: Action Unit based Depression Weights for py-feat ---
# Weights are heuristic, aiming for a raw score around -1 to 1 after normalization.
# Positive values contribute to depression score, negative values detract.
# py-feat AU intensities are typically 0-5.
AU_DEPRESSION_WEIGHTS = {
    'AU01': 0.1,   # Inner Brow Raiser (Sadness component)
    'AU02': 0.05,  # Outer Brow Raiser (Surprise, Fear component - slight)
    'AU04': 0.2,   # Brow Lowerer (Anger, Sadness, Concentration)
    'AU05': 0.05,  # Upper Lid Raiser (Surprise, Fear - slight)
    'AU06': -0.3,  # Cheek Raiser (Happiness - strong anti-dep)
    'AU07': 0.15,  # Lid Tightener (Anger, Fear, Pain, Tension)
    'AU09': 0.1,   # Nose Wrinkler (Disgust)
    'AU10': 0.1,   # Upper Lip Raiser (Disgust, Sadness)
    'AU12': -0.4,  # Lip Corner Puller (Happiness - very strong anti-dep)
    'AU14': -0.05, # Dimpler (slight happiness/social)
    'AU15': 0.25,  # Lip Corner Depressor (Sadness - strong dep)
    'AU17': 0.2,   # Chin Raiser (Sadness, Disgust)
    'AU20': 0.05,  # Lip Stretcher (Fear, Sadness, Anger - slight)
    'AU23': 0.1,   # Lip Tightener (Anger, Contempt, Dislike)
    'AU24': 0.05,  # Lip Pressor (Anger, Frustration - slight)
    'AU25': -0.05, # Lips Part (Social engagement - slight anti-dep, but can be neutral)
    'AU26': 0.05,  # Jaw Drop (Surprise, Slackness - slight dep if slack)
    'AU43': 0.0,   # Eyes Closed (can be neutral or part of other expressions) - pyfeat might use AU45 for blink
    # Add more AUs if py-feat consistently reports them and literature supports.
    # Note: py-feat might output AUs like 'AU01_r' for intensity. We'll handle base AU names.
}
# Normalization factor for AU-based score. Sum of absolute weights ~2.0.
# Max intensity per AU (e.g. 5). If 3-4 AUs active, score could be e.g. 2.0 * 3 * avg_intensity(2.5) = 15.
# To get to ~[-1, 1], we need to divide by something like 7-10. Let's try a dynamic normalization.
# --- END NEW AU Weights ---

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
FACE_OVERALL_WEIGHT = 1 # Face contributes 30%
VOICE_OVERALL_WEIGHT = 0.0 # Voice (emotion + acoustics) contributes 30%
TEXT_OVERALL_WEIGHT = 0.0 # Text analysis contributes 40

# --- End Depression Analysis Configuration ---

# Preload/reuse DeepFace emotion model
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
        emotion_model="svm", # Keep emotion model loaded for now if needed later
        device='auto'
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
        # Add handling for Path objects if whisper returns them
        if isinstance(obj, pathlib.Path):
             return str(obj)
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
        data_to_dump["results"] = [] 

    try:
        with open(temp_result_path, 'w') as f:
            json.dump(data_to_dump, f, cls=EmotionEncoder)
    except Exception as e:
        print(f"Error updating progress file {temp_result_path}: {e}")

# Function to calculate raw depression metric using py-feat Action Units
def calculate_raw_depression_metric(face_au_data, voice_emotion_data, voice_acoustic_features):
    """
    Calculate a raw depression metric based on facial Action Units (AUs),
    voice emotion, and voice acoustic features.
    The score is roughly in [-1, 1].

    Args:
        face_au_data: Dict from py-feat, containing 'aus' dictionary and possibly 'error'.
                      e.g., {'second': 0, 'aus': {'AU01': 0.5, ...}, 'error': None}
        voice_emotion_data: Dict possibly containing 'dominant_emotion' and 'emotions'.
        voice_acoustic_features: Dict containing extracted acoustic features for the second.

    Returns:
        A raw score roughly between -1 and 1.
    """
    face_dep_contribution = 0.0
    voice_dep_contribution = 0.0
    face_data_available = False
    voice_data_available = False

    # --- Calculate Face Depression Contribution from AUs ---
    if face_au_data and 'aus' in face_au_data and isinstance(face_au_data['aus'], dict) and face_au_data['aus']:
        temp_face_score = 0.0
        num_aus_used_in_score = 0
        active_au_intensities_sum = 0.0 # Sum of intensities of AUs used in score

        for au_name_full, intensity in face_au_data['aus'].items():
            # py-feat might return AU01_r, AU01_c. We want to use the base AUxx.
            au_base_name = au_name_full.split('_')[0] # e.g., "AU01" from "AU01_r"

            if au_base_name in AU_DEPRESSION_WEIGHTS and isinstance(intensity, (float, int)) and intensity > 0: # Only consider active AUs
                weight = AU_DEPRESSION_WEIGHTS[au_base_name]
                temp_face_score += intensity * weight
                num_aus_used_in_score += 1
                active_au_intensities_sum += intensity

        if num_aus_used_in_score > 0:
            # Normalize the score.
            # Simple normalization: divide by sum of active AU intensities if that sum is not tiny,
            # This makes it an average weighted contribution per unit of AU intensity.
            # Or, divide by a fixed factor related to expected max sum of AU intensities.
            # Let's try normalizing by a factor related to the number of AUs and their typical max intensity.
            # Max possible intensity sum from pyfeat is variable.
            # Let's scale it. If an average AU intensity is ~2.5 on a 0-5 scale.
            # And average weight magnitude is ~0.15.
            # Score per AU = 2.5 * 0.15 = 0.375. If 3-4 AUs are active: score ~1.1 to 1.5.
            # This seems a reasonable starting point for temp_face_score.
            # Let's cap and scale.
            # Max possible sum of weights is around 2. Typical max intensity is 5.
            # Max score before normalization could be sum_abs_weights * max_intensity = ~2 * 5 = 10
            # Normalization_factor could be e.g. 5.0 to bring it into +/- 2 range, then clip.
            # For a score between -1 and 1:
            # If we divide by (num_aus_used_in_score * average_max_intensity_per_au * average_abs_weight)
            # Let's try a simpler normalization first: scale by a fixed factor.
            # If max score is ~10, to get to ~1, divide by 10.
            # If max score is ~2 (if weights are small), divide by 2.
            # The sum of absolute values of weights is:
            # 0.1+0.05+0.2+0.05+0.3+0.15+0.1+0.1+0.4+0.05+0.25+0.2+0.05+0.1+0.05+0.05+0.05 = 2.3
            # If average intensity is 2.5, then temp_face_score could average num_aus * 2.5 * (avg_weight)
            # If num_aus_used_in_score is, say, 3, and average intensity is 2.5:
            # Effective denominator for normalization: num_aus_used_in_score * 2.5 (assuming weights average to bring it to [-1,1])
            # A simple scaling factor:
            SCALING_FACTOR_AU = 2.5 # Heuristic: chosen to bring score toward [-1, 1]
                                   # Assumes a few AUs active with moderate intensity
            
            if num_aus_used_in_score > 0 : # Avoid division by zero if somehow active_au_intensities_sum is 0
                # Normalize by the number of AUs that contributed to the score,
                # scaled by a factor to bring it into the desired [-1, 1] range.
                # This means the contribution is an "average" per active AU.
                face_dep_contribution = temp_face_score / (num_aus_used_in_score * SCALING_FACTOR_AU)
            else:
                face_dep_contribution = 0.0

            # Clip to ensure it's within [-1, 1] as other scores
            face_dep_contribution = max(-1.0, min(1.0, face_dep_contribution))
            face_data_available = True
        else: # No relevant AUs active
            face_dep_contribution = 0.0
            # If there were AUs but none in our map, or all zero, still considered data
            face_data_available = True 

    elif face_au_data and face_au_data.get('error'):
        if face_au_data.get('error') == 'no face detected':
            face_dep_contribution = FACE_DEPRESSION_WEIGHTS.get('no face detected', 0.0)
        else: # other errors
            face_dep_contribution = 0.0 # Or some other general error score
        face_data_available = True
    else: # No AU data dictionary, no error (e.g. py-feat returned empty or unexpected)
        face_dep_contribution = 0.0
        face_data_available = False # Mark as no data if py-feat output was not usable

    # --- Calculate Voice Depression Contribution (Combined Emotion + Acoustics) ---
    # This part is largely reused from the previous 'calculate_per_second_raw_depression_metric'
    voice_emotion_contribution = 0.0
    voice_acoustic_contribution = 0.0
    emotion_data_used = False
    acoustic_data_used = False

    # 1. Contribution from Voice Emotion Category
    voice_emotions = voice_emotion_data.get('emotions', {})
    voice_dominant = voice_emotion_data.get('dominant_emotion')

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
                scaling_params and scaling_params['scale'] != 0):
                # Scale to roughly [-1, 1] if params are good
                scaled_value = (feature_value - scaling_params['center']) / scaling_params['scale']
                scaled_value = max(-1.5, min(1.5, scaled_value)) # Clip extreme values after scaling
                temp_acoustic_score += scaled_value * weight
                num_acoustic_features_used += 1
        
        if num_acoustic_features_used > 0:
            voice_acoustic_contribution = temp_acoustic_score / num_acoustic_features_used # Average contribution
            acoustic_data_used = True
        else: # No valid acoustic features processed
            voice_acoustic_contribution = 0.0
    else: # Error in acoustic features or no features
        # Check for specific error types if needed, e.g. from voice_acoustic_features.get('error')
        voice_acoustic_contribution = 0.0

    # Combine voice emotion and acoustic contributions
    if emotion_data_used and acoustic_data_used:
        voice_dep_contribution = (VOICE_EMOTION_WEIGHT * voice_emotion_contribution +
                                  VOICE_ACOUSTIC_WEIGHT * voice_acoustic_contribution)
    elif emotion_data_used:
        voice_dep_contribution = voice_emotion_contribution # Only emotion data
    elif acoustic_data_used:
        voice_dep_contribution = voice_acoustic_contribution # Only acoustic data
    else: # No usable voice data
        voice_dep_contribution = 0.0
    
    if emotion_data_used or acoustic_data_used:
        voice_data_available = True

    # --- Combine Face and Voice Contributions ---
    # Weighted sum, only include components if data was available
    total_weight = 0
    combined_score = 0
    
    # Current weights are FACE_OVERALL_WEIGHT, VOICE_OVERALL_WEIGHT (TEXT is separate)
    # These weights seem to be for the final score, not this raw per-second metric.
    # For this raw metric, let's assume equal weighting if both are available,
    # or full weight to one if only one is available.
    # This part needs to be consistent with how FACE_OVERALL_WEIGHT and VOICE_OVERALL_WEIGHT are used later.
    # For now, let's use a simple average if both available.

    if face_data_available and voice_data_available:
        # This assumes face_dep_contribution and voice_dep_contribution are already scaled ~[-1,1]
        # The overall weights (FACE_OVERALL_WEIGHT, VOICE_OVERALL_WEIGHT) will be applied later.
        # Here, we are just calculating a combined raw score for this second from face/voice.
        # Let's keep them separate for now and let the main analysis loop combine them with overall weights.
        # So, this function should ideally return face_dep_contribution and voice_dep_contribution separately
        # OR return a combined score based on some internal weighting logic.
        # The original function returned a single raw score.
        # Let's stick to that for now: simple average if both present.
        combined_score = (face_dep_contribution + voice_dep_contribution) / 2
        # If one is missing, the other takes full weight. This is implicitly handled if one is 0.
        # But if we want to be explicit:
        # if not voice_data_available: combined_score = face_dep_contribution
        # if not face_data_available: combined_score = voice_dep_contribution

    elif face_data_available:
        combined_score = face_dep_contribution
    elif voice_data_available:
        combined_score = voice_dep_contribution
    else: # Neither face nor voice data available for this second
        combined_score = 0.0 # Or some other neutral/default

    return combined_score # This is the raw per-second score based on face/voice

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
    face_au_results = [] # Changed from face_results_per_second to store AU data
    voice_results_per_second = []
    acoustic_features_per_second = []
    # combined_results_per_second = [] # This was not used, can be removed
    raw_scores_per_second = []
    final_results_dict = { # Renamed from final_results to avoid conflict with update_progress arg
        "status": "processing", "progress": 0, "message": "Initializing...",
        "total_seconds": 0, "overall_depression_score": 0,
        "transcript": None, "transcript_error": None,
        "text_depression_analysis": None,
        "results": []
    }
    # temp_result_path = os.path.join(analysis_dir, f"{result_id}_progress.json") # Not needed if global update_progress used

    # Local update_progress removed, using global one:
    # update_progress(result_id, status, progress, message, total_seconds, results_data)

    audio_path = None # Define audio_path earlier

    try:
        # Check if models are loaded
        if face_detector is None:
            raise ValueError("py-feat face_detector is not initialized.")
        if voice_feature_extractor is None or voice_model is None:
            raise ValueError("Voice emotion model failed to load.")

        # Use global update_progress
        # Note: total_seconds (duration) isn't known yet. Can pass 0 or update later.
        # Let's calculate duration first.
        
        # Initial progress update
        # update_progress(result_id, "processing", 1, "Opening video file...", 0) # Old style

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
             raise IOError(f"Cannot open video file: {video_path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        final_results_dict["total_seconds"] = int(duration) # Store duration
        if duration == 0:
             raise ValueError("Video duration is zero or could not be determined.")

        # Now we have duration, update progress properly
        update_progress(result_id, "processing", 1, "Opening video file...", int(duration), final_results_dict)


        update_progress(result_id, "processing", 5, "Extracting audio...", int(duration), final_results_dict)
        audio_path = extract_audio_from_video(video_path, output_dir=analysis_dir)
        if not audio_path:
            final_results_dict["message"] = "Audio extraction failed. Skipping voice and text analysis."
            update_progress(result_id, "processing", 10, final_results_dict["message"], int(duration), final_results_dict)
            # Continue with face analysis only

        transcript = None
        # transcript_error = None # Already part of final_results_dict
        if audio_path and whisper_model:
            update_progress(result_id, "processing", 10, "Transcribing audio (may take time)...", int(duration), final_results_dict)
            transcript, transcript_error_msg = transcribe_audio(audio_path) # Returns two values
            final_results_dict["transcript"] = transcript
            final_results_dict["transcript_error"] = transcript_error_msg
            if transcript_error_msg:
                 final_results_dict["message"] = f"Transcription failed: {transcript_error_msg}. Proceeding without text analysis."
                 update_progress(result_id, "processing", 15, final_results_dict["message"], int(duration), final_results_dict)
            else:
                 update_progress(result_id, "processing", 15, "Transcription complete.", int(duration), final_results_dict)
        elif not audio_path:
            final_results_dict["transcript_error"] = "No audio extracted"
            update_progress(result_id, "processing", 15, "Skipping transcription (no audio).", int(duration), final_results_dict)
        else: 
            final_results_dict["transcript_error"] = "Whisper model not loaded"
            update_progress(result_id, "processing", 15, "Skipping transcription (model load failed).", int(duration), final_results_dict)

        text_analysis_result = None
        if transcript and text_depression_pipeline:
            update_progress(result_id, "processing", 20, "Analyzing text for depression...", int(duration), final_results_dict)
            text_analysis_result = analyze_text_depression(transcript)
            final_results_dict["text_depression_analysis"] = text_analysis_result
            update_progress(result_id, "processing", 25, "Text analysis complete.", int(duration), final_results_dict)
        elif not transcript:
            final_results_dict["text_depression_analysis"] = {"error": "No transcript available"}
            update_progress(result_id, "processing", 25, "Skipping text analysis (no transcript).", int(duration), final_results_dict)
        else: 
            final_results_dict["text_depression_analysis"] = {"error": "Text depression model not loaded"}
            update_progress(result_id, "processing", 25, "Skipping text analysis (model load failed).", int(duration), final_results_dict)

        # --- Face Analysis Loop (py-feat) ---
        # Progress for face analysis: 30% to 50%
        base_face_progress = 30
        face_progress_range = 20 
        update_progress(result_id, "processing", base_face_progress, "Analyzing facial action units...", int(duration), final_results_dict)
        
        frame_interval = int(fps) if fps > 0 else 1
        current_frame = 0
        second = 0
        
        while current_frame < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = video.read()
            if not ret: break

            current_face_au_data = {'second': second, 'aus': {}, 'error': None}
            temp_image_path = None # Initialize outside try

            try:
                if face_detector is None: # Should have been caught earlier, but good practice
                    raise RuntimeError("py-feat face_detector is not initialized.")

                # --- Resize Frame to 720p (Maintain Aspect Ratio) ---
                target_height = 720
                h, w = frame.shape[:2]
                if h > target_height:
                    ratio = target_height / h
                    target_width = int(w * ratio)
                    frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame
                
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # --- py-feat WORKAROUND: Save numpy array to temp file --- 
                # Ensure tempfile is imported globally if not already
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=analysis_dir) as temp_f:
                    temp_image_path = temp_f.name
                cv2.imwrite(temp_image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                
                detected_faces = face_detector.detect_image(temp_image_path) # Using detect_image
                
                if not detected_faces.empty and 'aus' in detected_faces.columns: # Check if 'aus' column exists
                    # Access the first face's AU data (assuming single face focus for now)
                    # detected_faces.aus is a DataFrame itself for AUs. Get first row as Series.
                    first_face_aus_series = detected_faces.aus.iloc[0] 
                    current_face_au_data['aus'] = {au: float(val) for au, val in first_face_aus_series.items()}
                else:
                    current_face_au_data['error'] = "no face detected"
            
            except Exception as e:
                print(f"Error processing face with py-feat in frame {current_frame}: {e}")
                current_face_au_data['error'] = f"analysis error: {str(e)}"
            finally:
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except Exception as e_remove:
                        print(f"Error removing temp image {temp_image_path}: {e_remove}")
            
            face_au_results.append(current_face_au_data)

            # Update progress - Face analysis part (e.g., 30% to 50%)
            progress_percent = base_face_progress + int(((second + 1) / duration * face_progress_range) if duration > 0 else 0)
            update_progress(result_id, "processing", progress_percent, f"Analyzing facial action units... ({second+1}/{int(duration)}s)", int(duration), final_results_dict)
            
            second += 1
            current_frame += frame_interval
        video.release()

        # --- Voice Analysis (Emotion + Acoustic Features) ---
        # Progress for voice: 50% to 85%
        base_voice_progress = base_face_progress + face_progress_range # Starts after face
        voice_progress_range = 35

        if audio_path and voice_model:
            update_progress(result_id, "processing", base_voice_progress, "Analyzing voice emotions and features...", int(duration), final_results_dict)
            voice_results_per_second, acoustic_features_per_second = analyze_audio_by_second_hf(
                audio_path, duration, result_id, # Pass result_id for global update_progress
                start_progress=base_voice_progress, 
                progress_range=voice_progress_range,
                total_duration_for_progress=int(duration) # Pass total_duration
            )
        elif not audio_path:
             update_progress(result_id, "processing", base_voice_progress + voice_progress_range, "Skipping voice analysis (no audio extracted)...", int(duration), final_results_dict)
             voice_results_per_second = [{'second': s, 'dominant_emotion': 'no audio extracted', 'emotions': {}} for s in range(int(duration))]
             acoustic_features_per_second = [{'second': s, 'error': 'No audio extracted'} for s in range(int(duration))]
        else: 
             update_progress(result_id, "processing", base_voice_progress + voice_progress_range, "Skipping voice analysis (model load failed)...", int(duration), final_results_dict)
             voice_results_per_second = [{'second': s, 'dominant_emotion': 'model error', 'emotions': {}} for s in range(int(duration))]
             acoustic_features_per_second = [{'second': s, 'error': 'Model load failed'} for s in range(int(duration))]
        
        # --- Combine Per-Second Results and Calculate Raw Scores ---
        # Progress: 85% to 90%
        base_combine_progress = base_voice_progress + voice_progress_range
        combine_progress_range = 5
        update_progress(result_id, "processing", base_combine_progress, "Combining per-second results...", int(duration), final_results_dict)

        for i in range(int(duration)):
            # Get face data for the current second (now from face_au_results)
            current_face_data = next((item for item in face_au_results if item['second'] == i), {'second': i, 'aus': {}, 'error': 'no data'})
            current_voice_data = next((item for item in voice_results_per_second if item['second'] == i), {'second': i, 'dominant_emotion': 'no data', 'emotions': {}})
            current_acoustic_data = next((item for item in acoustic_features_per_second if item['second'] == i), {'second': i, 'error': 'no data'})
            
            # Calculate raw score for this second
            raw_score = calculate_raw_depression_metric(current_face_data, current_voice_data, current_acoustic_data)
            raw_scores_per_second.append(raw_score)
            
            # Append detailed data for this second to final_results_dict['results']
            final_results_dict['results'].append({
                'second': i,
                'face_au_data': current_face_data, # Store AU data
                'voice_emotion_data': current_voice_data,
                'voice_acoustic_features': current_acoustic_data,
                'raw_depression_score_second': raw_score
            })
            
            progress_percent = base_combine_progress + int(((i + 1) / duration * combine_progress_range) if duration > 0 else 0)
            update_progress(result_id, "processing", progress_percent, f"Combining results... ({i+1}/{int(duration)}s)", int(duration), final_results_dict)

        # --- Calculate Overall Depression Score ---
        # Progress: 90% to 95%
        base_overall_score_progress = base_combine_progress + combine_progress_range
        overall_score_progress_range = 5
        update_progress(result_id, "processing", base_overall_score_progress, "Calculating overall score...", int(duration), final_results_dict)

        overall_raw_score = np.mean([s for s in raw_scores_per_second if s is not None]) if raw_scores_per_second else 0.0
        
        # Text analysis contribution
        text_depression_numeric_score = DEFAULT_TEXT_DEPRESSION_SCORE 
        if text_analysis_result and not text_analysis_result.get("error"):
            # Assuming text_analysis_result is like: {"label": "moderate", "score": 0.6 ...}
            # or list of [{"label": "not depression", "score": ...}, {"label": "moderate", ...}]
            # We need to map this to a single score, e.g., using TEXT_DEPRESSION_LABEL_TO_SCORE
            
            processed_text_score = 0.0
            if isinstance(text_analysis_result, list): # Handle pipeline output
                # Find the highest probability label or a specific one like 'depression'
                # For DepRoBERTa, it gives 'not depression', 'moderate', 'severe'
                # We want to turn this into a score from 0 to 1 (higher = more depression)
                temp_scores = {r['label'].lower(): r['score'] for r in text_analysis_result}
                if 'severe' in temp_scores and temp_scores['severe'] > 0.5: # Example threshold
                    processed_text_score = TEXT_DEPRESSION_LABEL_TO_SCORE.get('severe', 1.0)
                elif 'moderate' in temp_scores and temp_scores['moderate'] > 0.5:
                    processed_text_score = TEXT_DEPRESSION_LABEL_TO_SCORE.get('moderate', 0.5)
                else: # Predominantly 'not depression' or low scores for others
                    processed_text_score = TEXT_DEPRESSION_LABEL_TO_SCORE.get('not depression', 0.0)
            elif isinstance(text_analysis_result, dict) and 'label' in text_analysis_result: # Simpler dict case
                 processed_text_score = TEXT_DEPRESSION_LABEL_TO_SCORE.get(text_analysis_result['label'].lower(), DEFAULT_TEXT_DEPRESSION_SCORE)
            text_depression_numeric_score = processed_text_score

        # Combine overall_raw_score (from face/voice) with text_depression_numeric_score
        # Ensure scores are roughly in a compatible range (e.g. text_depression_numeric_score is 0-1)
        # overall_raw_score is ~[-1, 1]. Let's scale it to [0, 1] for easier combination.
        # (overall_raw_score + 1) / 2 would map [-1, 1] to [0, 1]
        scaled_audiovisual_score = (overall_raw_score + 1) / 2.0

        # Weighted combination
        # FACE_OVERALL_WEIGHT and VOICE_OVERALL_WEIGHT might need re-evaluation
        # if overall_raw_score is already a combined audio-visual score.
        # Let's assume overall_raw_score is the combined AV score.
        # The weights should be: AV_WEIGHT and TEXT_WEIGHT.
        # Let AV_WEIGHT = FACE_OVERALL_WEIGHT + VOICE_OVERALL_WEIGHT
        # However, the existing FACE_OVERALL_WEIGHT = 0.0, VOICE_OVERALL_WEIGHT = 1.7
        # TEXT_OVERALL_WEIGHT = 0.3
        # This suggests face is not used or py-feat is meant to provide emotions to be weighted like deepface.
        # Given FACE_OVERALL_WEIGHT = 0.0, this implies face AUs might not be used in final score
        # or the weights need adjustment for AU-based scores.
        # For now, using the structure:
        # combined_score = AV_part * AV_WEIGHT + TEXT_part * TEXT_WEIGHT
        # Let's assume VOICE_OVERALL_WEIGHT is for the scaled_audiovisual_score.

        final_weighted_score = (scaled_audiovisual_score * VOICE_OVERALL_WEIGHT) + \
                               (text_depression_numeric_score * TEXT_OVERALL_WEIGHT)
        
        # Normalize by sum of weights if they don't sum to 1, to keep score in expected range
        total_applied_weight = VOICE_OVERALL_WEIGHT + TEXT_OVERALL_WEIGHT
        if total_applied_weight > 1e-6 : # Avoid division by zero
            final_weighted_score = final_weighted_score / total_applied_weight
        else: # Should not happen with current weights
            final_weighted_score = scaled_audiovisual_score # Fallback if text weight is zero

        # Sigmoid scaling unless bypassed
        if not bypass_sigmoid:
            # Adjusted sigmoid: k affects steepness, x0 is midpoint
            # This maps a wider range of raw scores to 0-100
            # Current final_weighted_score is likely 0-1.
            # If sigmoid is desired, apply to map e.g. 0-1 -> 0-100 more dynamically.
            # Original sigmoid was: final_sigmoid = 1 / (1 + np.exp(-k * (weighted_score - x0)))
            # Let's assume final_weighted_score is already the value to be scaled to 0-100.
            # So, multiply by 100.
            overall_final_score = np.clip(final_weighted_score * 100, 0, 100)
        else: # Bypass: just clip the raw weighted score (0-1 range) and scale to 0-100
            overall_final_score = np.clip(final_weighted_score * 100, 0, 100)

        final_results_dict["overall_depression_score"] = overall_final_score
        final_results_dict["status"] = "completed"
        final_results_dict["message"] = f"Analysis complete. Overall Depression Score: {overall_final_score:.2f}"
        
        update_progress(result_id, "completed", 100, final_results_dict["message"], int(duration), final_results_dict)

    except Exception as e:
        print(f"Error in analyze_video_emotions (result_id: {result_id}): {e}")
        error_message = f"Error during analysis: {str(e)}"
        # Ensure final_results_dict is used for update_progress
        final_results_dict["status"] = "error"
        final_results_dict["message"] = error_message
        # Populate results with any partial data if available
        # final_results_dict["results"] = combined_results_per_second # combined_results_per_second was removed
        
        # If duration was calculated, use it, otherwise use 0
        current_duration = final_results_dict.get("total_seconds", 0)
        update_progress(result_id, "error", 100, error_message, current_duration, final_results_dict)

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError as remove_err:
                print(f"Error removing temporary audio file {audio_path}: {remove_err}")
        # Clean up the entire analysis_dir if error and no useful results, or after successful processing?
        # For now, manual cleanup or timed cleanup will handle it.


# Make sure analyze_audio_by_second_hf is adapted to use global update_progress
# and has access to result_id and total_duration if it's to call it.

def analyze_audio_by_second_hf(audio_path, duration, result_id, start_progress=0, progress_range=40, total_duration_for_progress=None):
    """Processes audio in 1-second segments for voice emotion and acoustic features."""
    voice_results = []
    acoustic_features_results = []

    # Local update_audio_progress function removed. Using global update_progress.
    # The necessary parameters (result_id, total_duration_for_progress) are passed to this function.

    if total_duration_for_progress is None: # Fallback if not provided
        total_duration_for_progress = int(duration)

    try:
        audio = AudioSegment.from_file(audio_path)
        target_sr = voice_feature_extractor.sampling_rate if voice_feature_extractor else 16000

        for second in range(int(duration)):
            start_ms = second * 1000
            end_ms = (second + 1) * 1000
            if end_ms > len(audio): break # Ensure segment is within audio length
            segment = audio[start_ms:end_ms]

            current_acoustic_features = {
                'rms_mean': None, 'rms_std': None, 'spectral_centroid_mean': None,
                'spectral_centroid_std': None, 'zcr_mean': None, 'zcr_std': None,
                'pitch_mean': None, 'pitch_std': None, 'mfcc_mean_std': None, 'error': None
            }
            emotion_result = {'dominant_emotion': 'pending', 'emotions': {}}
            temp_path = None

            try:
                # Ensure tempfile is imported globally
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                # Export segment to temp file
                segment.set_channels(1).set_frame_rate(target_sr).export(temp_path, format='wav')
                
                y, sr = librosa.load(temp_path, sr=target_sr)

                if np.abs(y).sum() > 1e-5: # If segment is not effectively silent
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
                        else: current_acoustic_features['pitch_mean']=0.0; current_acoustic_features['pitch_std']=0.0 # Use float
                    except Exception as fe: print(f"Pitch Err {second}: {fe}"); current_acoustic_features['pitch_mean']=0.0; current_acoustic_features['pitch_std']=0.0
                    try: mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13); current_acoustic_features['mfcc_mean_std']=np.std(np.mean(mfccs, axis=1))
                    except Exception as fe: print(f"MFCC Err {second}: {fe}")
                else: # Segment is silent or near-silent
                    current_acoustic_features['error'] = "silent segment"
                    # Set numerical features to neutral values (e.g., 0 or typical means)
                    for key in ['rms_mean', 'rms_std', 'spectral_centroid_mean', 'spectral_centroid_std', 
                                'zcr_mean', 'zcr_std', 'pitch_mean', 'pitch_std', 'mfcc_mean_std']:
                        if key not in current_acoustic_features or current_acoustic_features[key] is None:
                             current_acoustic_features[key] = 0.0


                emotion_result = predict_voice_emotion_hf(temp_path)
            
            except Exception as extract_err:
                print(f"Error processing audio segment {second}: {extract_err}")
                current_acoustic_features['error'] = str(extract_err)
                emotion_result = {'dominant_emotion': 'analysis error', 'emotions': {}}
            finally:
                if temp_path and os.path.exists(temp_path):
                    try: os.unlink(temp_path) # Use unlink for temp files
                    except OSError as unlink_err: print(f"Error deleting temp audio file {temp_path}: {unlink_err}")

            voice_results.append({'second': second, **emotion_result})
            acoustic_features_results.append({'second': second, **current_acoustic_features})

            # Call global update_progress
            current_progress_val = start_progress + int(((second + 1) / duration * progress_range) if duration > 0 else 0)
            progress_message = f"Analyzing voice emotions & features... ({second+1}/{int(duration)}s)"
            update_progress(result_id, "processing", current_progress_val, progress_message, total_duration_for_progress, None) # results=None during processing

        return voice_results, acoustic_features_results

    except Exception as e:
        print(f"Major error in analyze_audio_by_second_hf (result_id: {result_id}): {e}")
        duration_int = int(duration) if duration else 0
        default_emotion = [{"second": s, "dominant_emotion": "analysis error", "emotions": {}} for s in range(duration_int)]
        default_acoustic = [{"second": s, "error": f"Overall audio analysis failed: {str(e)}"} for s in range(duration_int)]
        # Update progress to indicate error in this stage if possible
        error_progress_val = start_progress + progress_range # Mark this stage as 'done' but with error
        update_progress(result_id, "error", error_progress_val, f"Voice analysis failed: {str(e)}", total_duration_for_progress, None)
        return default_emotion, default_acoustic

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