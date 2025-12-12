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
cat_encoder = LabelEncoder()
df["category"] = cat_encoder.fit_transform(df["category"])

# ------------------------------
# Model: Category Classifier
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"],
    test_size=0.2,
    random_state=42,
    stratify=df["category"]
)

vectorizer_cat = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer_cat.fit_transform(X_train)
X_test_vec = vectorizer_cat.transform(X_test)

clf_cat = LogisticRegression(max_iter=300)
clf_cat.fit(X_train_vec, y_train)

print("Category Model Accuracy:", clf_cat.score(X_test_vec, y_test))

os.makedirs("models", exist_ok=True)
pickle.dump(clf_cat, open("models/model_category.pkl", "wb"))
pickle.dump(vectorizer_cat, open("models/vectorizer_category.pkl", "wb"))
joblib.dump(cat_encoder, "models/category_encoder.pkl")

# --- Inspect model predictions on test set ---
y_pred = clf_cat.predict(X_test_vec)
# Build comparison table
results = pd.DataFrame({
    "text": X_test.values,
    "true_category": cat_encoder.inverse_transform(y_test),
    "predicted_category": cat_encoder.inverse_transform(y_pred)
})

# Filter misclassified rows
mistakes = results[results["true_category"] != results["predicted_category"]]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print("=== MISCLASSIFIED CATEGORY PREDICTIONS ===")
print(mistakes)
print(f"\nTotal mistakes: {len(mistakes)} out of {len(results)}")

print("\nTraining complete! Models saved in /models/")

