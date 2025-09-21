import os
import pandas as pd
import numpy as np
import re
import nltk
import torch
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import wandb
from kaggle_secrets import UserSecretsClient

# ----------------- NLTK Setup -----------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+',' ', text)
    text = re.sub(r'[^a-z\s]',' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop and len(t) > 1]
    return " ".join(tokens)

# ----------------- Load Datasets -----------------
# Update paths according to your data folder
folders = [
    "data/archive (4)",
    "data/archive (5)",
    "data/archive (6)"
]

dfs = []
for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        df = pd.read_csv(path)
        if 'title' in df.columns and 'text' in df.columns:
            df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
        else:
            df.rename(columns={df.columns[0]: 'content'}, inplace=True)
        # Map label to 0/1
        if 'label' in df.columns and df['label'].dtype == object:
            df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
        dfs.append(df[['content', 'label']])
df_all = pd.concat(dfs, ignore_index=True)
df_all['clean'] = df_all['content'].apply(clean_text)

# ----------------- Train-Test Split -----------------
X = df_all['clean']
y = df_all['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------- Prepare HuggingFace Dataset -----------------
train_df = pd.DataFrame({'text': X_train.tolist(), 'label': y_train.tolist()})
test_df  = pd.DataFrame({'text': X_test.tolist(), 'label': y_test.tolist()})

hf_train = Dataset.from_pandas(train_df)
hf_test = Dataset.from_pandas(test_df)

# ----------------- Tokenization -----------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

hf_train = hf_train.map(tokenize_fn, batched=True)
hf_test  = hf_test.map(tokenize_fn, batched=True)

hf_train.set_format('torch')
hf_test.set_format('torch')
hf_train = hf_train.remove_columns(['text'])
hf_test  = hf_test.remove_columns(['text'])

# ----------------- W&B Setup -----------------
secrets = UserSecretsClient()
wandb_api_key = secrets.get_secret("WANDB_API_KEY")  # replace with your secret name
if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("✅ Logged in to Weights & Biases successfully.")
else:
    print("⚠ WANDB_API_KEY not found. W&B logging disabled.")
    os.environ["WANDB_DISABLED"] = "true"

# ----------------- Trainer Arguments -----------------
args = TrainingArguments(
    output_dir='./checkpoints/distilbert-fake-news',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=1,
    report_to="wandb"
)

accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# ----------------- Model -----------------
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=hf_train,
    eval_dataset=hf_test,
    compute_metrics=compute_metrics
)

# ----------------- Train & Evaluate -----------------
trainer.train()
trainer.evaluate()

# ----------------- Save Model & Tokenizer -----------------
output_dir = "checkpoints/distilbert-fake-news-final"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model and tokenizer saved to {output_dir}")