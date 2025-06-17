import optuna
import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import TranslationDataset
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
DATA_PATH = "data/client1.json"

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    max_length = trial.suggest_categorical("max_length", [64, 96, 128])

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

    for name, param in model.named_parameters():
        if "model.decoder.final_logits_bias" not in name and "model.shared" not in name:
            param.requires_grad = False

    dataset = TranslationDataset(DATA_PATH, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        break  

    return loss.item() 

if __name__ == "__main__":
    print("Starting HPO with Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("\n✅ HPO Complete")
    print("Best hyperparameters:")
    for key, val in study.best_params.items():
        print(f"{key}: {val}")

    os.makedirs("hpo_results", exist_ok=True)
    with open("hpo_results/best_params.txt", "w") as f:
        for key, val in study.best_params.items():
            f.write(f"{key}: {val}\n")
    print("\n📁 Saved best parameters to hpo_results/best_params.txt")

    with open("hpo_results/best_params.py", "w") as f:
        f.write("best_params = {\n")
        for key, val in study.best_params.items():
            f.write(f"    '{key}': {val},\n")
        f.write("}\n")
    print("📁 Saved best parameters to hpo_results/best_params.py")
