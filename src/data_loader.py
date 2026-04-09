from datasets import load_dataset


def load_imdb_dataset():
    dataset = load_dataset("imdb")
    return dataset


if __name__ == "__main__":
    dataset = load_imdb_dataset()
    print(dataset)
