import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

save_path = "checkpoints/distilbert-fake-news-final"

tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_label(text, model, tokenizer, device, max_length=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
    return "REAL" if pred_id==1 else "FAKE"

sample_text = "This is a test article about a political event."
print("Sample Text:", sample_text)
print("Predicted Label:", predict_label(sample_text, model, tokenizer, device))
