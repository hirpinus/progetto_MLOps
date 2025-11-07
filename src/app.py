from flask import Flask, request, jsonify
from sentiment_predict import SentimentPredictor
from pathlib import Path

app = Flask(__name__)

BASE_PATH = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_PATH / "models"
predictor = SentimentPredictor(str(MODEL_PATH))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = predictor.predict(text)
    return jsonify({
        'sentiment': result['sentiment'],
        'confidence': result['confidence']
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)