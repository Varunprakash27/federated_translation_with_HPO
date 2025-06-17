import json
from torch.utils.data import Dataset
from transformers import MarianTokenizer

class TranslationDataset(Dataset):
    def __init__(self, data_path, tokenizer, source_lang="de", target_lang="en", max_length=64):
        self.tokenizer = tokenizer
        self.data = self.load_data(data_path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

    def load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data[idx][self.source_lang]
        target_text = self.data[idx][self.target_lang]

        tokenized = self.tokenizer.prepare_seq2seq_batch(
            src_texts=[source_text],
            tgt_texts=[target_text],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized['input_ids'].squeeze(),
            "attention_mask": tokenized['attention_mask'].squeeze(),
            "labels": tokenized['labels'].squeeze()
        }