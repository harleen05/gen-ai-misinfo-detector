from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import base64
import io
import warnings
from PIL import Image
import librosa
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import lime
from lime import lime_text
from lime.lime_image import LimeImageExplainer
import google.generativeai as genai
import joblib
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)

# ------------------------
# Gemini client
# ------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None
    print("[WARN] GEMINI_API_KEY not set. /explain will return an error.")

# ------------------------
# Explanation Engine (Gemini)
# ------------------------
class ExplanationLayerOSS:
    CONFIDENCE_MAP = {
        (0.0, 0.4): "low chance of being fake",
        (0.4, 0.7): "moderate chance of being fake",
        (0.7, 1.0): "high chance of being fake"
    }

    TECHNIQUE_LIBRARY = {
        "audio": {
            "robotic tone": "AI voice synthesis often produces slightly monotone or robotic speech.",
            "frequency artifacts": "Deepfake voices can leave behind unusual frequency spikes.",
            "unnatural pauses": "Synthetic voices sometimes pause in ways that don’t match human breathing."
        },
        "text": {
            "emotional language": "Fake news often exaggerates emotions to manipulate the reader.",
            "clickbait": "Overly dramatic headlines are designed to grab attention, not inform.",
            "repetitive phrases": "AI-generated or fake content often repeats words unnaturally."
        },
        "visual": {
            "blurry edges": "AI-generated images often fail to render sharp edges.",
            "lighting mismatch": "Fake composites show inconsistent lighting between objects.",
            "distorted hands/faces": "Deepfakes sometimes struggle with human anatomy."
        }
    }

    def _confidence_label(self, score: float) -> str:
        for (low, high), label in self.CONFIDENCE_MAP.items():
            if low <= score <= high:
                return label
        return "uncertain"

    def _reason_parser(self, model_output):
        modality = model_output.get("modality")
        score = model_output.get("score", 0.0)
        features = model_output.get("features", [])

        feature_explanations = []
        for f in features:
            extra = self.TECHNIQUE_LIBRARY.get(modality, {}).get(f, "")
            feature_explanations.append(f"{f} ({extra})" if extra else f)

        return f"{modality.upper()} model flagged: {', '.join(feature_explanations)}. Confidence={score:.2f} ({self._confidence_label(score)})."

    def generate_explanation(self, model_outputs, mode="general"):
        if isinstance(model_outputs, dict):
            model_outputs = [model_outputs]

        reasons = [self._reason_parser(out) for out in model_outputs]
        combined_reasons = "\n".join(reasons)

        prompt = f"""
        You are an AI misinformation educator.
        Evidence from detection models:
        {combined_reasons}

        TASK:
        - Translate this into a {mode}-level explanation
        - Cover:
          1. Why it may be misleading
          2. What visible signs a normal user could spot
          3. Practical steps to verify
          4. Educational tip to avoid similar tricks in future
        Output as clear structured JSON with keys:
        why_misleading, visible_signs, verification_steps, educational_tip, summary
        """

        if gemini_model is None:
            return {
                "verdict": "suspicious",
                "confidence_summary": [self._confidence_label(out.get("score", 0.0)) for out in model_outputs],
                "technical_reasons": reasons,
                "llm_explanation": "Gemini API key not configured; LLM explanation unavailable."
            }

        try:
            response = gemini_model.generate_content(prompt)
            llm_text = response.text
        except Exception as e:
            llm_text = f"LLM explanation call failed: {str(e)}"

        return {
            "verdict": "suspicious",
            "confidence_summary": [self._confidence_label(out.get("score", 0.0)) for out in model_outputs],
            "technical_reasons": reasons,
            "llm_explanation": llm_text
        }

engine = ExplanationLayerOSS()

@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json(force=True)
        outputs = data.get("model_outputs", [])
        mode = data.get("mode", "general")

        if not outputs:
            return jsonify({"error": "model_outputs is required and cannot be empty"}), 400

        result = engine.generate_explanation(outputs, mode)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------
# MODEL LOADING HELPERS
# ------------------------
def load_pickle_model(path):
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f"[INFO] Loaded pickle model from {path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {str(e)}")
    else:
        print(f"[WARN] Pickle model path not found: {path}")
    return None

def load_keras_model(path, input_shape=(224, 224, 3), num_classes=2):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            print(f"[INFO] Loaded Keras model from {path}")
            return model
        except Exception as e:
            print(f"[WARN] {path} not a full model, trying weights-only: {str(e)}")
            try:
                base = tf.keras.applications.EfficientNetB0(
                    include_top=False, input_shape=input_shape, pooling="avg"
                )
                x = tf.keras.layers.Dense(num_classes, activation="softmax")(base.output)
                model = tf.keras.Model(inputs=base.input, outputs=x)
                model.load_weights(path)
                print(f"[INFO] Loaded EfficientNetB0 weights from {path}")
                return model
            except Exception as e2:
                print(f"[ERROR] Could not load {path}: {str(e2)}")
    else:
        print(f"[WARN] Keras model path not found: {path}")
    return None

# Define same architecture used during training
class TextClassifier(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim=128, num_classes=2):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_torch_model(path, input_dim=5000, hidden_dim=128, num_classes=2, map_location="cpu"):
    if os.path.exists(path):
        try:
            model = TextClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
            state_dict = torch.load(path, map_location=map_location)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[INFO] Loaded Torch model from {path}")
            return model
        except Exception as e:
            print(f"[ERROR] Could not load state_dict from {path}: {str(e)}")
    else:
        print(f"[WARN] Torch model path not found: {path}")
    return None

def load_audio_file(path, target_sr=16000):
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"[WARN] Librosa failed, falling back to soundfile: {e}")
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
    return y, sr

# ------------------------
# LOAD MODELS (REQUIRED)
# ------------------------
os.makedirs("modeltext/checkpoints", exist_ok=True)
os.makedirs("visual", exist_ok=True)
os.makedirs("audio/checkpoints", exist_ok=True)

# text: assume class index 0 = real, 1 = fake (common convention)
TEXT_MODEL_PATH = "modeltext/checkpoints/model.pkl"
TEXT_VECTORIZER_PATH = "modeltext/checkpoints/vectorizer.pkl"

# image: assume class index 0 = real, 1 = fake
VISUAL_MODEL_PATH = "visual/vit_deepfake_visual.h5"

# audio: your original code assumed pred == 1 -> real; otherwise fake
AUDIO_MODEL_PATH = "audio/checkpoints/best_model.pkl"

text_model = load_torch_model(TEXT_MODEL_PATH)
text_vectorizer = load_pickle_model(TEXT_VECTORIZER_PATH)
visual_model = load_keras_model(VISUAL_MODEL_PATH)
audio_model = load_pickle_model(AUDIO_MODEL_PATH)

if text_vectorizer is None:
    # Only as a placeholder to avoid attribute errors; endpoints will still error if not fitted.
    text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    print("[WARN] text_vectorizer.pkl missing; a new TfidfVectorizer was created but NOT fitted. /check/text will likely fail.")

# ------------------------
# EXPLANATION HELPERS
# ------------------------
def generate_text_explanation(text, model, vectorizer, prediction_prob):
    try:
        if model is None or vectorizer is None:
            return {"method": "LIME", "error": "Text model or vectorizer not loaded"}

        explainer = lime_text.LimeTextExplainer(class_names=["Real", "Fake"])

        def predict_proba_fn(texts):
            feats = vectorizer.transform(texts).toarray()
            x = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

        exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)
        explanation_data = []
        for word, importance in exp.as_list():
            explanation_data.append({
                "word": word,
                "importance": float(importance),
                "contribution": "supports_fake" if importance > 0 else "supports_real"
            })
        return {
            "method": "LIME",
            "top_features": sorted(explanation_data, key=lambda x: abs(x['importance']), reverse=True)[:10]
        }
    except Exception as e:
        return {"method": "LIME", "error": f"Text explanation error: {str(e)}"}

def generate_image_explanation(image_array, model, prediction_prob):
    try:
        # Placeholder: LIME image explainer could be added here later
        img = image_array[0] if len(image_array.shape) == 4 else image_array
        h, w, c = img.shape
        return {
            "method": "Statistical Analysis",
            "dimensions": f"{w}x{h}x{c}",
            "brightness_mean": float(np.mean(img))
        }
    except Exception as e:
        return {"error": f"Image explanation error: {str(e)}"}

def generate_audio_explanation(audio_features, model, audio_duration, prediction_prob):
    try:
        return {
            "method": "Audio Feature Analysis",
            "duration_seconds": audio_duration,
            "feature_count": int(audio_features.shape[1]) if len(audio_features.shape) == 2 else int(len(audio_features))
        }
    except Exception as e:
        return {"error": f"Audio explanation error: {str(e)}"}

# ------------------------
# ROUTES
# ------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Deepfake Detection API", "version": "1.0"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "text": text_model is not None,
            "text_vectorizer": text_vectorizer is not None,
            "visual": visual_model is not None,
            "audio": audio_model is not None,
            "gemini": gemini_model is not None
        }
    })

# ------------------------
# TEXT CHECK
# ------------------------
@app.route("/check/text", methods=["POST"])
def check_text():
    if text_model is None or text_vectorizer is None:
        return jsonify({"error": "Text detection model or vectorizer not loaded"}), 500

    data = request.get_json(force=True)
    text = data.get("content", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        feats = text_vectorizer.transform([text]).toarray()
    except Exception as e:
        return jsonify({"error": f"Text vectorizer not fitted or incompatible: {str(e)}"}), 500

    try:
        x = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            logits = text_model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
    except Exception as e:
        return jsonify({"error": f"Text prediction failed: {str(e)}"}), 500

    # Assumption: class index 1 = fake, 0 = real
    verdict = "fake" if pred_idx == 1 else "real"

    explanation = generate_text_explanation(text, text_model, text_vectorizer, confidence)
    return jsonify({
        "type": "text",
        "verdict": verdict,
        "confidence": confidence,
        "raw_probs": probs.tolist(),
        "prediction_index": pred_idx,
        "explanation": explanation
    })

# ------------------------
# IMAGE CHECK
# ------------------------
@app.route("/check/image", methods=["POST"])
def check_image():
    if visual_model is None:
        return jsonify({"error": "Visual deepfake model not loaded"}), 500

    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    try:
        if img_b64.startswith("data:image"):
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    img_resized = img.resize((224, 224))
    img_arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    try:
        pred = visual_model.predict(img_arr)[0]
        pred_idx = int(np.argmax(pred))
        confidence = float(pred[pred_idx])
    except Exception as e:
        return jsonify({"error": f"Image prediction failed: {str(e)}"}), 500

    # Assumption: class index 1 = fake, 0 = real
    verdict = "fake" if pred_idx == 1 else "real"

    explanation = generate_image_explanation(img_arr, visual_model, confidence)
    return jsonify({
        "type": "image",
        "verdict": verdict,
        "confidence": confidence,
        "raw_probs": pred.tolist(),
        "prediction_index": pred_idx,
        "explanation": explanation
    })

# ------------------------
# AUDIO CHECK
# ------------------------
@app.route("/check/audio", methods=["POST"])
def check_audio():
    if audio_model is None:
        return jsonify({"error": "Audio deepfake model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    temp_path = f"/tmp/{file.filename}"
    try:
        file.save(temp_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500

    try:
        y, sr = load_audio_file(temp_path, target_sr=16000)
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": f"Failed to read audio file: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if y is None or len(y) == 0:
        return jsonify({"error": "Empty audio signal"}), 400

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    try:
        if hasattr(audio_model, "predict_proba"):
            class_proba = audio_model.predict_proba(features)[0]
            pred_idx = int(np.argmax(class_proba))
            confidence = float(class_proba[pred_idx])
        else:
            # If the model doesn't support predict_proba, just get the label
            pred_idx = int(audio_model.predict(features)[0])
            confidence = 1.0  # unknown, treat as max-conf since no probs
    except Exception as e:
        return jsonify({"error": f"Audio prediction failed: {str(e)}"}), 500

    # Your original logic: pred == 1 -> real, else fake
    verdict = "real" if pred_idx == 1 else "fake"

    explanation = generate_audio_explanation(features, audio_model, float(len(y) / sr), float(confidence))
    return jsonify({
        "type": "audio",
        "verdict": verdict,
        "confidence": float(confidence),
        "duration": float(len(y) / sr),
        "prediction_index": pred_idx,
        "raw_probs": class_proba.tolist() if 'class_proba' in locals() else None,
        "explanation": explanation
    })

# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    # In production, set debug=False
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)