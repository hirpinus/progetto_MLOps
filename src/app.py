import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, make_asgi_app
from .sentiment_predict import SentimentPredictor
from pathlib import Path

# --- Modello Pydantic per la validazione dell'input ---
class TextIn(BaseModel):
    text: str

# --- Configurazione dell'applicazione FastAPI ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="API per l'analisi del sentiment con monitoraggio Prometheus.",
    version="1.0.0"
)

# --- Configurazione delle metriche Prometheus ---
PREDICTIONS_TOTAL = Counter(
    'sentiment_predictions_total',
    'Total number of sentiment predictions',
    ['sentiment']
)
PREDICTION_CONFIDENCE = Gauge(
    'prediction_confidence',
    'Confidence of the last sentiment prediction',
    ['sentiment']
)

# --- Caricamento del modello ---
model_path_env = os.getenv('MODEL_PATH', 'models')
if not Path(model_path_env).is_absolute():
    model_path = Path(__file__).resolve().parent.parent / model_path_env
else:
    model_path = Path(model_path_env)

print(f"Caricamento del modello dal percorso: {model_path}")
try:
    predictor = SentimentPredictor(str(model_path))
    print("Modello caricato con successo.")
except FileNotFoundError as e:
    print(f"Errore critico: {e}. Il modello non è stato trovato.")
    predictor = None

# --- Monta l'app delle metriche sull'endpoint /metrics ---
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# --- Definizione degli endpoint API ---
@app.post("/predict")
def predict(item: TextIn):
    """
    Esegue la predizione del sentiment su un testo fornito.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Modello non disponibile. Controlla i log del server.")

    text = item.text
    
    # Esegui la predizione
    result = predictor.predict(text)
    sentiment = result['sentiment']
    confidence = result['confidence']

    # Aggiorna le metriche Prometheus
    PREDICTIONS_TOTAL.labels(sentiment=sentiment).inc()
    PREDICTION_CONFIDENCE.labels(sentiment=sentiment).set(confidence)

    return result

@app.get("/health", summary="Controlla lo stato del servizio")
def health_check():
    """
    Endpoint di health check per verificare che il servizio e il modello siano attivi.
    """
    if predictor:
        return {"status": "ok", "model_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Modello non caricato.")

# Nota: L'avvio del server non viene più gestito qui,
# ma dal comando `uvicorn` nel Dockerfile.