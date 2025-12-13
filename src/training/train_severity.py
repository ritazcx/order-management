import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os
import joblib
from sentence_transformers import SentenceTransformer

# Load dataset
DATA_PATH = os.path.join("data", "train.csv")
df = pd.read_csv(DATA_PATH)
if len(df) == 0:
    raise ValueError("The training dataset is empty.")

required_columns = {"text", "severity"}
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

sev_encoder = LabelEncoder()
df["severity"] = sev_encoder.fit_transform(df["severity"])

# ------------------------------
# Model: Severity Classifier
# ------------------------------
try:
    # Using SentenceTransformer for better embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    X_emb = model.encode(df["text"].tolist(), show_progress_bar=True)
except Exception as e:
    raise RuntimeError(f"Error generating text embeddings: {e}")

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_emb, df["severity"],
    test_size=0.2,
    random_state=42,
    stratify=df["severity"]
)

clf_sev = LogisticRegression(max_iter=2000, random_state=42)
clf_sev.fit(X_train_s, y_train_s)

print("Severity Model - Train Accuracy:", clf_sev.score(X_train_s, y_train_s))
print("Severity Model - Test Accuracy:", clf_sev.score(X_test_s, y_test_s))

# # --- Inspect model predictions on test set ---
y_pred_s = clf_sev.predict(X_test_s)
print(classification_report(y_test_s, y_pred_s, target_names=sev_encoder.classes_))

# makesure models directory exists
os.makedirs("models", exist_ok=True)

# save the severity model and encoder
try:
    pickle.dump(clf_sev, open("models/model_severity.pkl", "wb"))
    joblib.dump(sev_encoder, "models/encoder_severity.pkl")

    model_info = {
        "sbert_model_name": "all-mpnet-base-v2",
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_accuracy": clf_sev.score(X_train_s, y_train_s),
        "test_accuracy": clf_sev.score(X_test_s, y_test_s)
    }
    with open("models/severity_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
except Exception as e:
    raise RuntimeError(f"Error saving severity model or encoder: {e}")

print("\nTraining complete! Models saved in /models/")

