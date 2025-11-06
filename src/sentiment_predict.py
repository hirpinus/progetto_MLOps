import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentPredictor:
    def __init__(self, model_path: str):
        self._check_tokenizer_assets(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _check_tokenizer_assets(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Cartella modello non trovata: {path}")
        files = set(os.listdir(path))
        has_tokenizer = (
            "tokenizer.json" in files or
            ("vocab.txt" in files) or
            ({"vocab.json", "merges.txt"} <= files)
        )
        if not has_tokenizer:
            raise FileNotFoundError(
                "File tokenizer mancanti in '{path}'. Aggiungi tokenizer.json oppure vocab.json + merges.txt (BPE) "
                "o vocab.txt (WordPiece). Ricorda di eseguire tokenizer.save_pretrained(output_dir) dopo il training."
                .replace("{path}", path)
            )

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
        return {"sentiment": sentiment_map.get(idx, str(idx)),
                "confidence": conf,
                "raw_scores": probs.tolist()}