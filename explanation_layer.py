from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
from typing import Dict, List, Union

# ------------------------
# Explanation Engine
# ------------------------
class ExplanationLayerOSS:
    """
    Advanced Explanation Engine (OSS):
    - Handles multimodal outputs (audio, text, visual)
    - Produces structured JSON
    - Provides fact-checking resource suggestions
    - Adaptive explanation style (student/general/expert)
    """

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

    FACT_CHECK_SOURCES = {
        "general": [
            {"name": "Google Fact Check Explorer", "url": "https://toolbox.google.com/factcheck/explorer"},
            {"name": "Snopes", "url": "https://www.snopes.com/"},
            {"name": "PolitiFact", "url": "https://www.politifact.com/"},
        ],
        "health": [
            {"name": "WHO Mythbusters", "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters"},
            {"name": "CDC Rumor Control", "url": "https://www.cdc.gov/coronavirus/2019-ncov/daily-life-coping/share-facts.html"},
        ],
        "visual": [
            {"name": "Google Reverse Image Search", "url": "https://images.google.com/"},
            {"name": "TinEye Reverse Image Search", "url": "https://tineye.com/"},
        ],
        "audio": [
            {"name": "Deepware Scanner", "url": "https://deepware.ai/"},
            {"name": "Reality Defender", "url": "https://realitydefender.ai/"},
        ]
    }

    def __init__(self, model_name: str = "llama3:latest"):
        self.model_name = model_name

    def _confidence_label(self, score: float) -> str:
        for (low, high), label in self.CONFIDENCE_MAP.items():
            if low <= score <= high:
                return label
        return "uncertain"

    def _reason_parser(self, model_output: Dict) -> str:
        modality = model_output.get("modality")
        score = model_output.get("score", 0.0)
        features: List[str] = model_output.get("features", [])

        feature_explanations = []
        for f in features:
            extra = self.TECHNIQUE_LIBRARY.get(modality, {}).get(f, "")
            feature_explanations.append(f"{f} ({extra})" if extra else f)

        return f"{modality.upper()} model flagged: {', '.join(feature_explanations)}. Confidence={score:.2f} ({self._confidence_label(score)})."

    def _counter_check(self, modality: str, text: str = "") -> List[Dict]:
        if modality == "text" and any(keyword in text.lower() for keyword in ["covid", "vaccine", "health", "virus"]):
            return self.FACT_CHECK_SOURCES["health"] + self.FACT_CHECK_SOURCES["general"]
        return self.FACT_CHECK_SOURCES.get(modality, []) + self.FACT_CHECK_SOURCES["general"]

    def generate_explanation(
        self, 
        model_outputs: Union[Dict, List[Dict]], 
        mode: str = "general"
    ) -> Dict:
        if isinstance(model_outputs, dict):
            model_outputs = [model_outputs]

        # Collect reasons
        reasons = [self._reason_parser(out) for out in model_outputs]
        combined_reasons = "\n".join(reasons)

        # Build OSS LLM prompt
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

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        llm_explanation = response["message"]["content"]

        # Counter-check links
        all_links = []
        for out in model_outputs:
            all_links.extend(self._counter_check(out.get("modality", ""), text=out.get("raw_text", "")))

        # Deduplicate links
        seen = set()
        counter_links = []
        for link in all_links:
            if link["url"] not in seen:
                counter_links.append(link)
                seen.add(link["url"])

        return {
            "verdict": "suspicious",
            "confidence_summary": [self._confidence_label(out.get("score", 0.0)) for out in model_outputs],
            "technical_reasons": reasons,
            "llm_explanation": llm_explanation,
            "counter_check_links": counter_links
        }


# ------------------------
# Flask API
# ------------------------
app = Flask(__name__)
CORS(app)

engine = ExplanationLayerOSS(model_name="llama3:latest")

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
