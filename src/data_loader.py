from datasets import load_dataset


def load_imdb_dataset():
    return load_dataset("imdb")


if __name__ == "__main__":
    dataset = load_imdb_dataset()

    print("Sample review:")
    print(dataset["train"][0]["text"])
    print("\nLabel:", dataset["train"][0]["label"])
