import json
import os

REGISTRY_PATH = "models/registry.json"

def load_latest_models():
    with open(REGISTRY_PATH, "r") as f:
        reg = json.load(f)

    return {
        "category_model": reg["category"]["latest"],
        "severity_model": reg["severity"]["latest"],
        "category_encoder": reg["category"]["encoder"],
        "severity_encoder": reg["severity"]["encoder"],
        "version": reg["version"]
    }
