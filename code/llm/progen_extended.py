import torch

from lib.model.progen.init_model import init_model
from lib.model.train import train
from lib.model.progen.init_tokenizer import init_tokenizer
from lib.data.datasets.GB1 import get_GB1_dataset

TEST_SPLIT = 0.1
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
N_EPOCHS = 1


def train_progen_extended():
    print("Loading tokenizer")
    tokenizer = init_tokenizer()
    tokenize = lambda sequence: torch.tensor(tokenizer.encode(sequence).ids)
    tokenize_batch = lambda sequences: [tokenize(sequence) for sequence in sequences]

    print("Loading GB1 data")
    train_sequences, train_fitnesses, test_sequences, test_fitnesses = get_GB1_dataset(
        tokenize=tokenize_batch, test_split=TEST_SPLIT
    )

    # TODO delete
    train_sequences = train_sequences[1:2]
    train_fitnesses = train_fitnesses[1:2]
    test_sequences = test_sequences[1:2]
    test_fitnesses = test_fitnesses[1:2]

    print("Connecting to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model")
    model = init_model(device=device)

    print("Training")
    train(
        model=model,
        device=device,

        train_data=train_sequences,
        train_labels=train_fitnesses,
        test_data=test_sequences,
        test_labels=test_fitnesses,

        loss_function=torch.nn.functional.l1_loss,

        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
    )


if __name__ == "__main__":
    train_progen_extended()
