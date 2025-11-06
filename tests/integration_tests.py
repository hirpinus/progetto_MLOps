import os, sys, subprocess, pytest

PROJECT_ROOT = "/workspaces/progetto_MLOps"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sentiment_predict import SentimentPredictor

def _has_tokenizer_assets(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    return (
        "tokenizer.json" in files
        or ({"vocab.json","merges.txt"} <= files)
        or "vocab.txt" in files
        or any(f.endswith(".model") for f in files)
    )

def _has_model_weights(path: str) -> bool:
    files = set(os.listdir(path)) if os.path.isdir(path) else set()
    return "model.safetensors" in files or "pytorch_model.bin" in files

@pytest
def test_model_assets_exist():
    if not os.path.isdir(MODEL_DIR):
        pytest.skip(f"Cartella modello assente: {MODEL_DIR}")
    assert os.path.isfile(os.path.join(MODEL_DIR, "config.json")), "config.json mancante"
    assert _has_model_weights(MODEL_DIR), "pesi del modello mancanti (model.safetensors o pytorch_model.bin)"
    assert _has_tokenizer_assets(MODEL_DIR), "asset tokenizer mancanti (tokenizer.json o vocab.*)"

@pytest
def test_can_load_and_predict():
    if not (_has_tokenizer_assets(MODEL_DIR) and _has_model_weights(MODEL_DIR)):
        pytest.skip("Asset modello/tokenizer mancanti; salto inferenza.")
    predictor = SentimentPredictor(MODEL_DIR)
    out = predictor.predict("Questo prodotto è fantastico!")
    assert set(["sentiment","confidence","raw_scores"]).issubset(out.keys())
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["raw_scores"], list)
    assert len(out["raw_scores"]) == predictor.model.config.num_labels

@pytest
def test_main_script_smoke_runs():
    if not (_has_tokenizer_assets(MODEL_DIR) and _has_model_weights(MODEL_DIR)):
        pytest.skip("Asset modello/tokenizer mancanti; salto smoke test main.")
    proc = subprocess.run(
        ["python", os.path.join(SRC_DIR, "main.py")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60
    )
    assert proc.returncode == 0, f"main.py ha fallito: {proc.stderr}"
    assert "Sentimento:" in proc.stdout