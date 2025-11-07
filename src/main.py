from sentiment_predict import SentimentPredictor
from pathlib import Path

def main():

    BASE_PATH = Path(__file__).resolve().parents[1]
    MODEL_PATH = BASE_PATH / "models"
    
    predictor = SentimentPredictor(str(MODEL_PATH))
    
    test_texts = [
        "Questo prodotto è fantastico!",
        "Non mi è piaciuto per niente.",
        "Il servizio clienti è stato molto professionale."
    ]

    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nTesto: {text}")
        print(f"Sentimento: {result['sentiment']}")
        print(f"Confidenza: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()