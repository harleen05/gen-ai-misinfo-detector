Our project is built around a seamless, multi-stage workflow that ensures accuracy, speed, and explainability at every step.

1. User Encounter
The process begins in the browser, as soon as a user encounters text, image, video, or audio content they want to verify.

2. Chrome Extension Activation
With a single click, our lightweight Chrome extension is activated. It extracts the relevant content—no matter the modality—and securely forwards it to the cloud-based backend.

3. Multi-Modal AI Analysis
The backend engine orchestrates specialized AI models:Vision Transformer (ViT) for image and video frame deepfake detection
  --CNN models for audio authenticity (mel-spectrogram analysis)
  --Transformer models (DistilBERT) for text and fake news
  --Each model is fine-tuned for its specific task and uses advanced explainability features.

4. Explanation Layer
--All results are passed through an explanation pipeline:
--Confidence Mapping: Converts raw scores into intuitive Low/Medium/High trust levels.
--Feature Explainer & LLM (Ollama/GPT): Adds human-readable reasons and, optionally, fact-checking sources for transparency.

Outputs: Verdict, confidence, technical reasons, and plain-language explanation are packaged into a single JSON response.

5. User Feedback & Display
The Chrome extension immediately displays:
 --A verdict badge (🛡 Real/Safe or ⚠️ Fake/Risky)
 --A human-readable reasoning

Users are empowered to understand both the what and the why for every piece of content they review.

6. Continuous Learning
Results and user feedback can be optionally logged (Wt&B, secured) to improve the models, adapt to new attack patterns, and enhance explanations over time.

