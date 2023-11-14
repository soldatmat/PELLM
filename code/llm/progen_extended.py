import torch

from lib.model.progen.extended.init_model import init_model
from lib.model.progen.extended.model_states import save_model_states
from lib.model.train import train
from lib.model.progen.init_tokenizer import init_tokenizer
from lib.data.datasets.GB1 import get_GB1_dataset


TEST_SPLIT = 0.1
BATCH_SIZE = 11
LEARNING_RATE = 1e-3
N_EPOCHS = 1

LOAD = "/models/progen_extended_0.pt"
SAVE = "/models/progen_extended_tmp.pt"


def train_progen_extended(
    test_split=TEST_SPLIT,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    state_dict_path: str = None,
    save_to: str = None,
    absolute_paths=False,
):
    print("Connecting to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer")
    tokenizer = init_tokenizer()

    print("Loading GB1 data")
    tokenize = lambda sequence: tokenizer.encode(sequence).ids
    train_sequences, train_fitnesses, test_sequences, test_fitnesses = get_GB1_dataset(
        tokenize=tokenize,
        test_split=test_split,
        device=device,
    )

    print("Initializing model")
    model = init_model(
        device=device,
        state_dict_path=state_dict_path,
        absolute_path=absolute_paths,
    )

    print("Training")
    loss_history = train(
        model=model,
        device=device,
        train_data=train_sequences,
        train_labels=train_fitnesses,
        test_data=test_sequences,
        test_labels=test_fitnesses,
        loss_function=torch.nn.functional.l1_loss,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
    )

    if save_to is not None:
        save_model_states(
            model,
            save_to=save_to,
            absolute_path=absolute_paths,
        )

    return loss_history


if __name__ == "__main__":
    loss_history = train_progen_extended(
        save_to=SAVE,
        state_dict_path=LOAD,
    )
