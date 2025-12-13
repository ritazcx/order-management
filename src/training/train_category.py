import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import joblib

# Load dataset
DATA_PATH = os.path.join("data", "train.csv")
df = pd.read_csv(DATA_PATH)
if len(df) == 0:
    raise ValueError("The training dataset is empty.")

required_columns = {"text", "category"}
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# 显示类别分布
print("\n=== Category Distribution ===")
category_counts = df["category"].value_counts()
print(category_counts)

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

print(f"\nTraining samples: {len(X_train)} Test samples: {len(X_test)}")

vectorizer_cat = TfidfVectorizer(stop_words="english",
                                 max_features=5000,  # 限制特征数量
                                 ngram_range=(1, 2))   # 使用unigram和bigram
X_train_vec = vectorizer_cat.fit_transform(X_train)
X_test_vec = vectorizer_cat.transform(X_test)

clf_cat = LogisticRegression(max_iter=500,
                             random_state=42,
                             class_weight='balanced')  # 处理类别不平衡
clf_cat.fit(X_train_vec, y_train)

print("Category Model - Train Accuracy:", clf_cat.score(X_train_vec, y_train))
print("Category Model - Test Accuracy:", clf_cat.score(X_test_vec, y_test))

# 详细分类报告
y_pred = clf_cat.predict(X_test_vec)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=cat_encoder.classes_))

# 混淆矩阵
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=[f"True {c}" for c in cat_encoder.classes_],
                     columns=[f"Pred {c}" for c in cat_encoder.classes_])
print(cm_df)

# 保存模型和向量化器
os.makedirs("models", exist_ok=True)
pickle.dump(clf_cat, open("models/model_category.pkl", "wb"))
pickle.dump(vectorizer_cat, open("models/vectorizer_category.pkl", "wb"))
joblib.dump(cat_encoder, "models/encoder_category.pkl")

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
print(f"Total samples: {len(results)}")
print(f"Misclassified: {len(mistakes)} ({len(mistakes)/len(results)*100:.2f}%)")
if len(mistakes) > 0:
    error_by_class = mistakes.groupby("true_category").size()
    print("\nMisclassified by true category:")
    print(error_by_class)
print("\nTraining complete! Models saved in /models/")

