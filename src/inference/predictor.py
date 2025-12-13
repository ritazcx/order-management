import json
import pickle
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from src.inference.model_loader import load_latest_models

def _safe_load(path):
    """Try joblib.load then pickle.load; raise descriptive error on failure."""
    if path is None or not isinstance(path, str) or not path.strip():
        raise ValueError("Invalid model path provided.")
    try:
        return joblib.load(path)
    except Exception:
        try:
            return pickle.load(open(path, 'rb'))
        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {e}")

class Predictor:

    def __init__(self):
        self.registry = load_latest_models()

        # Load Category Artifacts
        cat_vec_path = self.registry.get("category_vectorizer")
        cat_model_path = self.registry.get("category_model")
        cat_label_path = self.registry.get("category_encoder")

        # Load Severity Artifacts
        sev_model_path = self.registry.get("severity_model")
        sev_label_path = self.registry.get("severity_encoder")
        sbert_model_name = self.registry.get("sbert_model_name")

        # Load category vectorizer, encoder, model
        try:
            self.category_vectorizer = _safe_load(cat_vec_path)
        except Exception as e:
            raise RuntimeError(f"Error loading category vectorizer: {e}") 

        try:
            self.category_model = _safe_load(cat_model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading category model: {e}")

        try:
            self.category_encoder = _safe_load(cat_label_path)
        except Exception as e:
            raise RuntimeError(f"Error loading category encoder: {e}")

        # Load severity model, encoder

        try:
            self.severity_model = _safe_load(sev_model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading severity model: {e}")

        try:
            self.severity_encoder = _safe_load(sev_label_path)
        except Exception as e:
            raise RuntimeError(f"Error loading severity encoder: {e}")    

        try: 
            self.sbert_model = SentenceTransformer(sbert_model_name)
        except Exception as e:
            raise RuntimeError(f"Error loading SBERT model: {e}")

        # Load label mappings
        self.categories = list(self.category_encoder.classes_)
        self.severity_map = {i: label for i, label in enumerate(self.severity_encoder.classes_)}


    def get_model_version(self):
        return self.registry.get('version', 'unknown')

    def predict(self, text: str):
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            # 正确的特征提取流程：
            # 1. 先用SBERT模型获取文本嵌入
            text_embedding = self.sbert_model.encode([text])
            
            # 2. 对类别分类使用TF-IDF向量化
            cat_features = self.category_vectorizer.transform([text])

             # 3. 严重性分类使用SBERT嵌入
            sev_features = text_embedding
            
        except Exception as e:
            raise RuntimeError(f"Error during feature extraction: {e}")

        # Predictions
        try:
            cat_pred = self.category_model.predict(cat_features)[0]
            sev_pred = self.severity_model.predict(sev_features)[0]

            # Confidence
            if hasattr(self.category_model, "predict_proba") and hasattr(self.severity_model, "predict_proba"):
                cat_prob = np.max(self.category_model.predict_proba(cat_features))
                sev_prob = np.max(self.severity_model.predict_proba(sev_features))
                final_conf = float(np.mean([cat_prob, sev_prob]))

            return {
                "category": self.categories[cat_pred] if cat_pred < len(self.categories) else "Unknown",
                "severity": self.severity_map[sev_pred] if sev_pred in self.severity_map else "Unknown",
                "confidence": final_conf
            }
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")
        
        

    def __del__(self):
        """Cleanup resources if needed."""
        if hasattr(self, 'sbert_model'):
            del self.sbert_model