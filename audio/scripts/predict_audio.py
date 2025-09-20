import argparse, torch
from common.utils import CNN, predict_file
from common.config import device

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
args = parser.parse_args()

model = CNN()
model.load_state_dict(torch.load("checkpoints/best_final_model.pth", map_location=device))
model.to(device)

label = predict_file(model, args.file)
print("Prediction:", label)
