from __future__ import annotations
import os, sys, subprocess, pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # root della repo
SRC_DIR = ROOT / "src"
MODEL_DIR = ROOT / "models"

#if str(SRC_DIR) not in sys.path:
#    sys.path.insert(0, str(SRC_DIR))

from sentiment_predict import SentimentPredictor

def _has_tokenizer_assets(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = {p.name for p in path.iterdir()}
    return (
        "tokenizer.json" in files
        or {"vocab.json", "merges.txt"} <= files
        or "vocab.txt" in files
        or any(f.endswith(".model") for f in files)
    )

def _has_model_weights(path: Path) -> bool:
    files = {p.name for p in path.iterdir()} if path.is_dir() else set()
    return "model.safetensors" in files or "pytorch_model.bin" in files

def test_model_assets_exist():
    if not MODEL_DIR.is_dir():
        pytest.skip(f"Cartella modello assente: {MODEL_DIR}")
    assert (MODEL_DIR / "config.json").is_file(), "config.json mancante"
    assert _has_model_weights(MODEL_DIR), "pesi del modello mancanti (model.safetensors o pytorch_model.bin)"
    assert _has_tokenizer_assets(MODEL_DIR), "asset tokenizer mancanti (tokenizer.json o vocab.*)"

def test_can_load_and_predict():
    if not (_has_tokenizer_assets(MODEL_DIR) and _has_model_weights(MODEL_DIR)):
        pytest.skip("Asset modello/tokenizer mancanti; salto inferenza.")
    predictor = SentimentPredictor(str(MODEL_DIR))
    out = predictor.predict("Questo prodotto è fantastico!")
    assert {"sentiment","confidence","raw_scores"} <= set(out)
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["raw_scores"], list)
    assert len(out["raw_scores"]) == predictor.model.config.num_labels

def test_main_script_smoke_runs():
    if not (_has_tokenizer_assets(MODEL_DIR) and _has_model_weights(MODEL_DIR)):
        pytest.skip("Asset modello/tokenizer mancanti; salto smoke test main.")
    proc = subprocess.run(
        ["python", str(SRC_DIR / "main.py")],
        cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60
    )
    assert proc.returncode == 0, f"main.py ha fallito: {proc.stderr}"
    assert "Sentimento:" in proc.stdout