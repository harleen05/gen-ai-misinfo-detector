import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Load model
with open("checkpoints/logreg_model.pkl", "rb") as f:
    clf = pickle.load(f)

# Load your test embeddings and labels (example)
X_test_emb = np.load("checkpoints/X_test_emb.npy")
y_test = np.load("checkpoints/y_test.npy")

y_pred = clf.predict(X_test_emb)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
