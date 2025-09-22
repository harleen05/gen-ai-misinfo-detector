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
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)

# Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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

        response = gemini_model.generate_content(prompt)

        return {
            "verdict": "suspicious",
            "confidence_summary": [self._confidence_label(out.get("score", 0.0)) for out in model_outputs],
            "technical_reasons": reasons,
            "llm_explanation": response.text
        }

engine = ExplanationLayerOSS()

@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json()
        outputs = data.get("model_outputs", [])
        mode = data.get("mode", "general")
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
            return joblib.load(path)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {str(e)}")
    return None

def load_keras_model(path, input_shape=(224,224,3), num_classes=2):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            print(f"[WARN] {path} not a full model, trying weights-only: {str(e)}")
            try:
                base = tf.keras.applications.EfficientNetB0(
                    include_top=False, input_shape=input_shape, pooling="avg"
                )
                x = tf.keras.layers.Dense(num_classes, activation="softmax")(base.output)
                model = tf.keras.Model(inputs=base.input, outputs=x)
                model.load_weights(path)
                return model
            except Exception as e2:
                print(f"[ERROR] Could not load {path}: {str(e2)}")
    return None

def load_torch_model(path, map_location="cpu"):
    if os.path.exists(path):
        try:
            model = torch.load(path, map_location=map_location)
            model.eval()
            return model
        except Exception as e:
            print(f"[ERROR] Could not load torch model {path}: {str(e)}")
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
# LOAD MODELS
# ------------------------
os.makedirs("modeltext/checkpoints", exist_ok=True)
os.makedirs("visual", exist_ok=True)
os.makedirs("audio/checkpoints", exist_ok=True)

text_model = load_torch_model("modeltext/checkpoints/model.pkl")
text_vectorizer = load_pickle_model("modeltext/checkpoints/vectorizer.pkl")
visual_model = load_keras_model("visual/vit_deepfake_visual.h5")
audio_model = load_pickle_model("audio/checkpoints/best_model.pkl")

if text_vectorizer is None:
    text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))

# ------------------------
# EXPLANATION HELPERS
# ------------------------
def generate_text_explanation(text, model, vectorizer, prediction_prob):
    try:
        if model and vectorizer:
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
        else:
            return {"method": "Statistical Analysis", "note": "Fallback explanation"}
    except Exception as e:
        return {"error": f"Text explanation error: {str(e)}"}

def generate_image_explanation(image_array, model, prediction_prob):
    try:
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
            "feature_count": len(audio_features[0])
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
            "audio": audio_model is not None
        }
    })

@app.route("/check/text", methods=["POST"])
def check_text():
    data = request.get_json()
    text = data.get("content", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    pred, proba = 0, 0.5
    if text_model and text_vectorizer:
        try:
            feats = text_vectorizer.transform([text]).toarray()
            x = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                logits = text_model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            proba = float(np.max(probs))
        except Exception as e:
            print(f"[ERROR] Text prediction failed: {e}")

    explanation = generate_text_explanation(text, text_model, text_vectorizer, proba)
    return jsonify({"type": "text", "verdict": "fake" if pred==1 else "real", "confidence": proba, "explanation": explanation})

@app.route("/check/image", methods=["POST"])
def check_image():
    data = request.get_json()
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400
    if img_b64.startswith("data:image"):
        img_b64 = img_b64.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    img_resized = img.resize((224,224))
    img_arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)
    pred_class, confidence = 0, 0.5
    if visual_model:
        pred = visual_model.predict(img_arr)[0]
        confidence = float(np.max(pred))
        pred_class = int(np.argmax(pred))
    explanation = generate_image_explanation(img_arr, visual_model, confidence)
    return jsonify({"type": "image", "verdict": "fake" if pred_class==1 else "real", "confidence": confidence, "explanation": explanation})

@app.route("/check/audio", methods=["POST"])
def check_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files["file"]
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)

    y, sr = load_audio_file(temp_path, target_sr=16000)
    os.remove(temp_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    pred, proba = 0, 0.5
    if audio_model:
        pred = audio_model.predict(features)[0]
        proba = audio_model.predict_proba(features)[0].max()

    explanation = generate_audio_explanation(features, audio_model, float(len(y)/sr), float(proba))
    return jsonify({
        "type": "audio",
        "verdict": "real" if pred == 1 else "fake",
        "confidence": float(proba),
        "duration": float(len(y) / sr),
        "explanation": explanation
    })

# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
