from datasets import load_dataset
import pandas as pd


def convert_to_csv():
    dataset = load_dataset("imdb")

    train_data = dataset["train"]
    test_data = dataset["test"]

    def transform(split):
        texts = split["text"]
        labels = split["label"]

        # Convert labels
        labels = ["positive" if l == 1 else "negative" for l in labels]

        return pd.DataFrame({"text": texts, "label": labels})

    train_df = transform(train_data)
    test_df = transform(test_data)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = convert_to_csv()

    train_df = train_df.sample(5000, random_state=42)
    test_df = test_df.sample(1000, random_state=42)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("CSV files created!")
