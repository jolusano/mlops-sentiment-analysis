import gradio as gr
from transformers import pipeline
from google.cloud import aiplatform
import csv
from datetime import datetime

# -------- CONFIG --------
PROJECT_ID = "740124399188"
REGION = "us-central1"
ENDPOINT_ID = "2668189814926344192"

# -------- LOAD MODELS --------
hf_model = pipeline("sentiment-analysis")

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)


# -------- PREDICTIONS --------
def predict_hf(text):
    result = hf_model(text, truncation=True, max_length=512)[0]
    return f"{result['label']} ({result['score']:.2f})"


def predict_automl(text):
    response = endpoint.predict(instances=[{"text": text}])
    prediction = response.predictions[0]

    classes = prediction["classes"]
    scores = prediction["scores"]

    max_index = scores.index(max(scores))

    return f"{classes[max_index]} ({scores[max_index]:.2f})"


def compare_models(text):
    hf_raw = hf_model(text, truncation=True, max_length=512)[0]
    hf_result = {"label": hf_raw["label"], "score": float(hf_raw["score"])}

    response = endpoint.predict(instances=[{"text": text}])
    prediction = response.predictions[0]

    classes = prediction["classes"]
    scores = prediction["scores"]
    max_index = scores.index(max(scores))

    automl_result = {"label": classes[max_index], "score": scores[max_index]}

    # Log prediction
    log_prediction(text, hf_result, automl_result)

    return (
        f"{hf_result['label']} ({hf_result['score']:.2f})",
        f"{automl_result['label']} ({automl_result['score']:.2f})",
    )


def log_prediction(text, hf_result, automl_result):
    with open("logs/predictions.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                text,
                hf_result["label"],
                hf_result["score"],
                automl_result["label"],
                automl_result["score"],
                datetime.now(),
            ]
        )


# -------- GRADIO UI --------
interface = gr.Interface(
    fn=compare_models,
    inputs=gr.Textbox(lines=3, placeholder="Enter a movie review..."),
    outputs=[
        gr.Text(label="HuggingFace Prediction"),
        gr.Text(label="AutoML Prediction"),
    ],
    title="Sentiment Analysis Comparison",
    description="Compare predictions from HuggingFace and AutoML models",
)

# -------- RUN APP --------
if __name__ == "__main__":
    interface.launch()
