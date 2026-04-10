from transformers import pipeline
from google.cloud import aiplatform

# -------- CONFIG --------
PROJECT_ID = "740124399188"
REGION = "us-central1"
ENDPOINT_ID = "2668189814926344192"

# -------- LOAD HF MODEL --------
hf_model = pipeline("sentiment-analysis")


def predict_hf(text):
    result = hf_model(text)[0]
    return {"label": result["label"], "score": float(result["score"])}


# -------- LOAD VERTEX ENDPOINT --------
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)


def predict_automl(text):
    response = endpoint.predict(instances=[{"text": text}])
    prediction = response.predictions[0]

    classes = prediction["classes"]
    scores = prediction["scores"]

    # Find index of highest score
    max_index = scores.index(max(scores))

    return {"label": classes[max_index], "score": scores[max_index]}


# -------- TEST BOTH --------
if __name__ == "__main__":
    samples = [
        "This movie was amazing!",
        "Worst movie I have ever seen",
        "It was okay, not great but not terrible",
        "I loved the acting and the story",
        "This was a complete waste of time",
    ]

    for text in samples:
        hf_result = predict_hf(text)
        automl_result = predict_automl(text)

        print("\n========================")
        print("Input:", text)
        print("HuggingFace:", hf_result)
        print("AutoML:", automl_result)
