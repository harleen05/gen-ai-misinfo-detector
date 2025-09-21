import torch, os
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from common.config import device, base_dirs, protocol_files, hyperparams
from common.utils import CNN, train_model, to_mel, spec_augment, load_labels
from collections import Counter
from tqdm import tqdm

label_map = load_labels(protocol_files)
audio_files = []
for base in base_dirs[:2]: 
    audio_files.extend(glob.glob(os.path.join(base, "*.flac")))

features, labels = [], []
for f in tqdm(audio_files):
    fname = os.path.basename(f)
    if fname not in label_map: continue
    waveform, sr = torchaudio.load(f)
    mel = to_mel(waveform, sr)
    mel = spec_augment(mel)
    mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(40,200)).squeeze(0)
    features.append(mel)
    labels.append(label_map[fname])

features = torch.stack(features)
labels = torch.tensor(labels)
print("Label distribution:", Counter(labels.tolist()))

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

model = CNN(**hyperparams)
final_model = train_model(model, train_dataset, val_dataset, **hyperparams)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.save(final_model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_final_model.pth"))
pkl_path = os.path.join(CHECKPOINT_DIR, "audio_model.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(final_model, f)

print(f"Model saved at {pkl_path}")