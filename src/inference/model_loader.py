import json
import os

REGISTRY_PATH = "models/registry.json"

def load_latest_models():
    with open(REGISTRY_PATH, "r") as f:
        reg = json.load(f)

    return {
        "category_vectorizer": reg["category"]["vectorizer"],
        "category_model": reg["category"]["model"],
        "category_encoder": reg["category"]["encoder"],
        "severity_model": reg["severity"]["model"],
        "severity_encoder": reg["severity"]["encoder"],
        "sbert_model_name": reg["severity"]["sbert_model_name"],
        "version": reg["version"]
    }
