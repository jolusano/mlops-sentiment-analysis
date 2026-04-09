from transformers import pipeline


def load_model():
    classifier = pipeline("sentiment-analysis")
    return classifier


def predict(text, model):
    result = model(text)[0]
    return result


if __name__ == "__main__":
    model = load_model()

    sample_text = "This movie was absolutely amazing!"
    result = predict(sample_text, model)

    print("Input:", sample_text)
    print("Prediction:", result)
