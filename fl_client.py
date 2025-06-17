import flwr as fl
import torch
from transformers import MarianTokenizer, MarianMTModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import TranslationDataset
from hpo_results.best_params import best_params

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"

class TranslationClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

        self.learning_rate = best_params["learning_rate"]
        self.batch_size = best_params["batch_size"]
        self.max_length = best_params["max_length"]

        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

        for name, param in self.model.named_parameters():
            if "model.decoder.final_logits_bias" not in name and "model.shared" not in name:
                param.requires_grad = False

        dataset = TranslationDataset(f"data/client{client_id}.json", self.tokenizer, max_length=self.max_length)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values() if val.requires_grad]

    def set_parameters(self, parameters):
        keys = [k for k, v in self.model.state_dict().items() if v.requires_grad]
        full_state_dict = self.model.state_dict()
        for k, v in zip(keys, parameters):
            full_state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(full_state_dict)

    def fit(self, parameters, config=None):
        print(f"[Client {self.client_id}] >> FIT STARTED")
        self.set_parameters(parameters)
        self.model.train()

        for batch in self.dataloader:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f"[Client {self.client_id}] Loss: {loss.item():.4f}")
            break

        torch.save(self.model.state_dict(), f"client{self.client_id}_model.pt")
        print(f"[Client {self.client_id}] >> FIT ENDED | Saved model")
        return self.get_parameters(), len(self.dataloader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        return 0.0, len(self.dataloader.dataset), {}

if __name__ == "__main__":
    client = TranslationClient(client_id=1)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
