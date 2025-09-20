import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dirs = [
    "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac",
    "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac",
    "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_eval/flac"
]

protocol_files = {
    "train": "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
}

hyperparams = {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "conv1_filters": 16,
    "conv2_filters": 32,
    "fc_size": 64,
    "dropout": 0.0
}
