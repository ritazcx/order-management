import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from .model_loader import load_latest_models

class Predictor:

    def __init__(self):
        self.registry = load_latest_models()

        # Load encoder
        self.category_encoder = pickle.load(open(self.registry["category_encoder"], "rb"))
        self.severity_encoder = pickle.load(open(self.registry["severity_encoder"], "rb"))

        # Load ML models
        self.category_model = pickle.load(open(self.registry["category_model"], "rb"))
        self.severity_model = pickle.load(open(self.registry["severity_model"], "rb"))

    def get_model_version(self):
        return self.registry

    def predict(self, text: str):

        # Embedding
        cat_encoded = self.category_encoder.transform([text])
        sev_encoded = self.severity_encoder.transform([text])

        # Predictions
        cat_pred = self.category_model.predict(cat_encoded)[0]
        sev_pred = self.severity_model.predict(sev_encoded)[0]

        # Confidence
        cat_prob = np.max(self.category_model.predict_proba(cat_encoded))
        sev_prob = np.max(self.severity_model.predict_proba(sev_encoded))

        final_conf = float(np.mean([cat_prob, sev_prob]))

        return {
            "category": self.categories[cat_pred],
            "severity": self.severity_map[sev_pred],
            "confidence": final_conf
        }
