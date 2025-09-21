## 🔄 Workflow

- **User Encounter**  
  Starts in the browser when a user finds text, image, video, or audio they want to verify.  

- **Chrome Extension Activation**  
  With one click, the lightweight extension extracts the content and securely forwards it to the cloud backend.  

- **Multi-Modal AI Analysis**  
  - **Vision Transformer (ViT):** Image & video frame deepfake detection  
  - **CNN (CRNN):** Audio authenticity via mel-spectrograms  
  - **Transformer (DistilBERT):** Fake news & text verification  
  Each model is fine-tuned for its task with explainability features.  

- **Explanation Layer**  
  - **Confidence Mapping:** Converts raw scores → Low / Medium / High trust levels  
  - **Feature Explainer + LLM (Ollama/GPT):** Adds human-readable reasons & fact-checking sources  
  - Outputs are packaged as a **single JSON response** with verdict, confidence, reasons, and explanations  

- **User Feedback & Display**  
  - Verdict badge: 🛡 Real / Safe or ⚠️ Fake / Risky  
  - Clear, human-readable reasoning  
  Users see *not just the verdict, but also why*.  

- **Continuous Learning**  
  Optional (secure & Wt&B) logging of results + feedback.  
  Models adapt to new attack patterns and improve explanations over time.  
