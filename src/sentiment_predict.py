from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentPredictor:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()  # Imposta il modello in modalità valutazione
        
    def predict(self, text):
        # Tokenizza il testo
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        # Fai la predizione
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Ottieni la classe predetta e la confidenza
        predicted_class = torch.argmax(predictions).item()
        confidence = predictions[0][predicted_class].item()
        
        # Mappa le classi alle etichette
        sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
        
        return {
            "sentiment": sentiment_map[predicted_class],
            "confidence": confidence,
            "raw_scores": predictions[0].tolist()
        }