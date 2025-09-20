import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os, glob

def to_mel(waveform, sr, n_mels=40):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=512, hop_length=160, win_length=400, n_mels=n_mels
    )(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_spec_db

def spec_augment(spec, freq_mask_param=27, time_mask_param=100, num_masks=1):
    spec = spec.clone()
    _, num_freq, num_time = spec.shape
    for _ in range(num_masks):
        freq_mask = torch.randint(0, freq_mask_param, (1,)).item()
        freq_start = torch.randint(0, max(1, num_freq - freq_mask), (1,)).item()
        spec[:, freq_start:freq_start+freq_mask, :] = 0
        time_mask = torch.randint(0, time_mask_param, (1,)).item()
        time_start = torch.randint(0, max(1, num_time - time_mask), (1,)).item()
        spec[:, :, time_start:time_start+time_mask] = 0
    return spec

def load_labels(protocol_files):
    label_map = {}
    for split, path in protocol_files.items():
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                flac_name = parts[1] + ".flac"
                label = 0 if parts[4] == "bonafide" else 1
                label = 1 - label  
                label_map[flac_name] = label
    return label_map

class CNN(nn.Module):
    def __init__(self, conv1_filters=16, conv2_filters=32, fc_size=64, dropout=0.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv2_filters*10*50, fc_size)
        self.fc2 = nn.Linear(fc_size,2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def train_model(model, train_dataset, val_dataset, batch_size=32, lr=0.001, epochs=10):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    writer = SummaryWriter(log_dir="runs/final_model")
    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        train_preds, train_labels = [], []
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs,1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        val_preds, val_labels_list = [], []
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs,1).cpu().numpy())
                val_labels_list.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)
        val_prec = precision_score(val_labels_list, val_preds, zero_division=0)
        val_rec = recall_score(val_labels_list, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.3f}, P: {val_prec:.3f}, R: {val_rec:.3f}, F1: {val_f1:.3f}")

        writer.add_scalar("Loss/train", train_loss/len(train_loader), epoch)
        writer.add_scalar("Loss/val", val_loss/len(val_loader), epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Precision/val", val_prec, epoch)
        writer.add_scalar("Recall/val", val_rec, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_final_model.pth")

        scheduler.step()
    writer.close()
    return model

def predict_file(model, filepath):
    model.eval()
    with torch.no_grad():
        waveform, sr = torchaudio.load(filepath)
        mel = to_mel(waveform, sr)
        mel = F.interpolate(mel.unsqueeze(0), size=(40,200)).squeeze(0)
        mel = mel.unsqueeze(0).to(next(model.parameters()).device)
        outputs = model(mel)
        pred = torch.argmax(outputs, dim=1).item()
        return "bonafide" if pred==1 else "spoof"