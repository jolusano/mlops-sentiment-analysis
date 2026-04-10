import pandas as pd
from transformers import pipeline
from google.cloud import aiplatform

# -------- CONFIG --------
PROJECT_ID = "740124399188"
REGION = "us-central1"
ENDPOINT_ID = "2668189814926344192"

# -------- LOAD DATA --------
df = pd.read_csv("data/test.csv")

# Optional: limit to 1000 rows
df = df.sample(1000, random_state=42)

# -------- LOAD MODELS --------
hf_model = pipeline("sentiment-analysis")

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)


def predict_hf(text):
    result = hf_model(text, truncation=True, max_length=512)[0]

    return result["label"].lower()


def predict_automl(text):
    response = endpoint.predict(instances=[{"text": text}])
    prediction = response.predictions[0]

    classes = prediction["classes"]
    scores = prediction["scores"]

    max_index = scores.index(max(scores))
    return classes[max_index]


# -------- EVALUATION --------
hf_correct = 0
automl_correct = 0

for i, row in df.iterrows():
    text = row["text"]
    true_label = row["label"].lower()

    hf_pred = predict_hf(text)
    automl_pred = predict_automl(text)

    if hf_pred == true_label:
        hf_correct += 1

    if automl_pred == true_label:
        automl_correct += 1

# -------- RESULTS --------
total = len(df)

print("\n=== RESULTS ===")
print(f"HuggingFace Accuracy: {hf_correct / total:.4f}")
print(f"AutoML Accuracy: {automl_correct / total:.4f}")
