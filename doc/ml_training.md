# ML Training Guide

This document explains how to train the machine-learning models used in the AI Ticket Automation System.
Two models are trained:

1. **Category Classification Model** — Predicts the type of ticket
2. **Severity Classification Model** — Predicts Low / Medium / High severity

Both models are lightweight (sklearn), fast to predict, and suitable for real-time orchestration with n8n.

---

## 1. Overview

The ML layer provides:

* Fast and deterministic predictions
* Low inference cost
* Structured JSON output for n8n
* High reliability (no hallucinations)

It is responsible for:

* Intent classification (ticket category)
* Severity prediction
* Determining whether an LLM is needed

The LLM only runs *after* ML decides that deeper reasoning is required.

---

## 2. Data Format

Training data is stored as a CSV file (`train.csv`) with the following columns:

| text                            | category            | severity |
| ------------------------------- | ------------------- | -------- |
| My VPN keeps disconnecting…     | VPN / Connectivity  | High     |
| Please grant me access…         | Access / Permission | Low      |
| Outlook is not receiving emails | Email Issue         | High     |

### Required Fields

* **text**: User ticket content
* **category**: One of the predefined ticket types
* **severity**: Low / Medium / High

### Suggested Categories

These categories can be expanded later:

* Network Issue
* Access / Permission
* Software Bug
* Hardware Problem
* Email Issue
* VPN / Connectivity
* Office / Excel Issue
* Other IT Support

Severity: `Low | Medium | High`

---

## 3. Example Training Data (Starter Dataset)

Below is a sample dataset you can directly use to bootstrap your first model:

```
text,category,severity
"My VPN keeps disconnecting when I try to join meetings.","VPN / Connectivity","High"
"I cannot access the company's shared folder. Permission denied.","Access / Permission","Medium"
"Outlook is not receiving new emails since last night.","Email Issue","High"
"My laptop is overheating and shutting down.","Hardware Problem","Medium"
"The Wi-Fi connection is unstable on the 3rd floor.","Network Issue","Medium"
"Excel freezes every time I open a file.","Office/Excel Issue","Low"
"The internal web portal returns a 500 error.","Software Bug","Medium"
"Please grant me access to the finance folder.","Access / Permission","Low"
"My mouse suddenly stopped working today.","Hardware Problem","Low"
"I cannot connect to the VPN when traveling.","VPN / Connectivity","High"
"Slack notifications are delayed.","Software Bug","Low"
"The printer on floor 2 is jammed again.","Hardware Problem","Low"
"Wi-Fi shows connected but no internet.","Network Issue","High"
"Please reset my password.","Access / Permission","Low"
"Chrome crashes when opening certain pages.","Software Bug","Low"
"Emails are taking too long to send.","Email Issue","Medium"
"My monitor flickers randomly.","Hardware Problem","Medium"
"VPN login fails with authentication error.","VPN / Connectivity","High"
"The network switch in server room is down.","Network Issue","High"
"Excel formulas are not recalculating properly.","Office/Excel Issue","Medium"
```

You can expand to 200–300 samples for better performance.

---

## 4. Training Script (`train.py`)

The following script trains both Category and Severity models using sklearn.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("train.csv")

# --------- Model A: Category Classifier ---------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42
)

vectorizer_cat = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer_cat.fit_transform(X_train)
X_test_vec = vectorizer_cat.transform(X_test)

clf_cat = LogisticRegression(max_iter=200)
clf_cat.fit(X_train_vec, y_train)

print("Category Model Accuracy:", clf_cat.score(X_test_vec, y_test))

pickle.dump(clf_cat, open("model_category.pkl", "wb"))
pickle.dump(vectorizer_cat, open("vectorizer_category.pkl", "wb"))


# --------- Model B: Severity Classifier ---------
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    df["text"], df["severity"], test_size=0.2, random_state=42
)

vectorizer_sev = TfididfVectorizer(stop_words="english")
X_train_s_vec = vectorizer_sev.fit_transform(X_train_s)
X_test_s_vec = vectorizer_sev.transform(X_test_s)

clf_sev = LogisticRegression(max_iter=200)
clf_sev.fit(X_train_s_vec, y_train_s)

print("Severity Model Accuracy:", clf_sev.score(X_test_s_vec, y_test_s))

pickle.dump(clf_sev, open("model_severity.pkl", "wb"))
pickle.dump(vectorizer_sev, open("vectorizer_severity.pkl", "wb"))
```

---

## 5. Running Training

Run:

```
python train.py
```

This will generate:

```
model_category.pkl
vectorizer_category.pkl
model_severity.pkl
vectorizer_severity.pkl
```

These files are used during inference.

---

## 6. Inference API (FastAPI)

Create `api.py`:

```python
from fastapi import FastAPI
import pickle

app = FastAPI()

clf_cat = pickle.load(open("model_category.pkl", "rb"))
vec_cat = pickle.load(open("vectorizer_category.pkl", "rb"))

clf_sev = pickle.load(open("model_severity.pkl", "rb"))
vec_sev = pickle.load(open("vectorizer_severity.pkl", "rb"))


@app.post("/predict")
def predict(data: dict):
    text = data["text"]

    category = clf_cat.predict(vec_cat.transform([text]))[0]
    severity = clf_sev.predict(vec_sev.transform([text]))[0]

    needs_llm = (severity == "High")

    return {
        "category": category,
        "severity": severity,
        "needs_llm": bool(needs_llm)
    }
```

Start server:

```
uvicorn api:app --reload --port 8000
```

You should now have:

```
POST http://localhost:8000/predict
```

---

## 7. Output JSON Schema

The predicted output will follow this format:

```json
{
  "category": "Network Issue",
  "severity": "High",
  "needs_llm": true
}
```

This JSON is consumed by the n8n workflow.

---

## 8. Next Steps

Continue to:

* `/docs/n8n_workflow.md` → Build the orchestration layer
* `/docs/llm_prompts.md` → Add LLM reasoning
* `/docs/mvp_guide.md` → End-to-end working demo

ML layer is now complete.

