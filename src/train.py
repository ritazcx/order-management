import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import os
import joblib

# Load dataset
DATA_PATH = os.path.join("data", "train.csv")
df = pd.read_csv(DATA_PATH)
cat_encoder = LabelEncoder()
sev_encoder = LabelEncoder()
df["category"] = cat_encoder.fit_transform(df["category"])
df["severity"] = sev_encoder.fit_transform(df["severity"])

# ------------------------------
# Model A: Category Classifier
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"],
    test_size=0.4,
    random_state=42,
    stratify=df["category"]
)

vectorizer_cat = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer_cat.fit_transform(X_train)
X_test_vec = vectorizer_cat.transform(X_test)

print("X_train_vec shape:", X_train_vec.shape)
print("X_test_vec shape:", X_test_vec.shape)

print("Train categories:", set(y_train))
print("Test categories:", set(y_test))


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

# ------------------------------
# Model B: Severity Classifier
# ------------------------------
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    df["text"], df["severity"],
    test_size=0.4,
    random_state=42,
    stratify=df["severity"]
)

vectorizer_sev = TfidfVectorizer(stop_words="english")
X_train_s_vec = vectorizer_sev.fit_transform(X_train_s)
X_test_s_vec = vectorizer_sev.transform(X_test_s)

clf_sev = LogisticRegression(max_iter=300)
clf_sev.fit(X_train_s_vec, y_train_s)

print("Severity Model Accuracy:", clf_sev.score(X_test_s_vec, y_test_s))

pickle.dump(clf_sev, open("models/model_severity.pkl", "wb"))
pickle.dump(vectorizer_sev, open("models/vectorizer_severity.pkl", "wb"))

# # --- Inspect model predictions on test set ---
# y_pred_s = clf_sev.predict(X_test_s_vec)
# true_labels = sev_encoder.inverse_transform(y_test_s)
# pred_labels = sev_encoder.inverse_transform(y_pred_s)
# print("\n=== SEVERITY PREDICTIONS ON TEST DATA ===")
# for i in range(len(X_test_s)):
#     print(f"Text: {X_test_s.iloc[i][:60]}...")
#     print(f"True: {true_labels[i]},  Pred: {pred_labels[i]}")
#     print("-----")

print("\nTraining complete! Models saved in /models/")

