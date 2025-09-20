import glob, os
from common.config import base_dirs, protocol_files
from common.utils import CNN, predict_file, load_labels
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = load_labels(protocol_files)

model = CNN()
model.load_state_dict(torch.load("checkpoints/best_final_model.pth", map_location=device))
model.to(device)
eval_dir = base_dirs[2]
eval_files = glob.glob(os.path.join(eval_dir, "*.flac"))

eval_preds, eval_labels_list = [], []
for f in tqdm(eval_files):
    fname = os.path.basename(f)
    label_true = label_map.get(fname)
    pred_label = predict_file(model, f)
    if label_true is not None:
        eval_labels_list.append(label_true)
        eval_preds.append(1 if pred_label=="bonafide" else 0)

print("Accuracy:", accuracy_score(eval_labels_list, eval_preds))
print("Precision:", precision_score(eval_labels_list, eval_preds))
print("Recall:", recall_score(eval_labels_list, eval_preds))
print("F1:", f1_score(eval_labels_list, eval_preds))
print("Confusion Matrix:\n", confusion_matrix(eval_labels_list, eval_preds))
print("Classification Report:\n", classification_report(eval_labels_list, eval_preds, target_names=["spoof","bonafide"]))
