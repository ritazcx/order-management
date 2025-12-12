import pandas as pd
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
sev_encoder = LabelEncoder()
df["severity"] = sev_encoder.fit_transform(df["severity"])

# ------------------------------
# Model: Severity Classifier
# ------------------------------
# Using SentenceTransformer for better embeddings
model = SentenceTransformer('all-mpnet-base-v2')
X_emb = model.encode(df["text"].tolist(), show_progress_bar=True)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_emb, df["severity"],
    test_size=0.2,
    random_state=42,
    stratify=df["severity"]
)
# We don't use TF-IDF here since its performance is low
# vectorizer_sev = TfidfVectorizer(stop_words="english")
# X_train_s_vec = vectorizer_sev.fit_transform(X_train_s)
# X_test_s_vec = vectorizer_sev.transform(X_test_s)
clf_sev = LogisticRegression(max_iter=2000)
clf_sev.fit(X_train_s, y_train_s)

print("Severity Model Accuracy:", clf_sev.score(X_test_s, y_test_s))

# # --- Inspect model predictions on test set ---
y_pred_s = clf_sev.predict(X_test_s)
print(classification_report(y_test_s, y_pred_s, target_names=["Low", "Medium", "High"]))


pickle.dump(clf_sev, open("models/model_severity.pkl", "wb"))
# pickle.dump(vectorizer_sev, open("models/vectorizer_severity.pkl", "wb"))
joblib.dump(sev_encoder, "models/severity_encoder.pkl")

print("\nTraining complete! Models saved in /models/")

