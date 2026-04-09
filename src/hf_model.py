from transformers import pipeline


def load_model():
    classifier = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier


def predict(text, model):
    result = model(text)[0]
    return {"label": result["label"], "score": float(result["score"])}


if __name__ == "__main__":
    model = load_model()

    samples = [
        "This movie was absolutely amazing!",
        "Worst film I have ever seen.",
        "It was okay, nothing special.",
    ]

    for text in samples:
        result = predict(text, model)
        print("\nInput:", text)
        print("Prediction:", result)
