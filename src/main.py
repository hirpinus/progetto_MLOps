from sentiment_predict import SentimentPredictor

def main():
    # Percorso del modello salvato
    MODEL_PATH = "/workspaces/progetto_MLOps/models"
    
    # Inizializza il predictor
    predictor = SentimentPredictor(MODEL_PATH)
    
    # Esempi di testi da analizzare
    test_texts = [
        "Questo prodotto è fantastico!",
        "Non mi è piaciuto per niente.",
        "Il servizio clienti è stato molto professionale."
    ]
    
    # Fai predizioni
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nTesto: {text}")
        print(f"Sentimento: {result['sentiment']}")
        print(f"Confidenza: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()