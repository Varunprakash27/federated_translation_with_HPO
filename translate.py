import torch
from transformers import MarianTokenizer, MarianMTModel

MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "client1_model.pt"

print("[*] Loading tokenizer and model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"[*] Loaded trained weights from {MODEL_PATH}")
except FileNotFoundError:
    print(f"[!] Trained model file '{MODEL_PATH}' not found. Using base pretrained model.")

model.eval()

def translate(text: str) -> str:
    batch = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        translated = model.generate(**batch)

    return tokenizer.decode(translated[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("⚙️ German to English Translator")
    print("-----------------------------------")
    while True:
        german = input("Enter German text (or type 'exit'): ").strip()
        if german.lower() == "exit":
            break
        english = translate(german)
        print(f"➡️  English translation: {english}\n")
