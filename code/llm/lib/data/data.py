from datasets import Dataset


def create_dataset(data, labels):
    dataset = Dataset.from_list(data)
    dataset = dataset.add_column("labels", labels)
    return dataset
