import torch

from lib.model.extended.init_model import init_model
from lib.model.progen.init_tokenizer import init_tokenizer
from lib.data.datasets.GB1 import get_GB1_dataset

N_INITIALIZATIONS = 100
N_DATA = 100

LOSS_FUNCTION = torch.nn.functional.mse_loss


def test_initialization(
    n_initializations: int,
    loss_function,
    n_data: int = None,
    data_indexes=None,
    return_data=False,
):
    print("Connecting to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer")
    tokenizer = init_tokenizer()

    print("Loading GB1 data")
    # TODO control test dataset
    tokenize = lambda sequence: tokenizer.encode(sequence).ids
    sequences, fitnesses = get_GB1_dataset(
        tokenize=tokenize,
        shuffle=data_indexes==None,
        n_data=n_data,
        data_indexes=data_indexes,
        device=device,
    )

    print("Initializing model")
    model = init_model().to(device)

    print("Running initialization test")
    losses = torch.empty(n_initializations).to(device)
    for init in range(n_initializations):
        model.fitness_head = torch.nn.Linear(
            in_features=1024,
            out_features=1,
            bias=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(sequences)
            losses[init] = loss_function(outputs, fitnesses)
            loss_print(losses[init], init + 1)

    if return_data:
        return losses, sequences, fitnesses
    return losses


def loss_print(loss, init_idx):
    print(f"Initialization {init_idx}: loss = {loss}")


if __name__ == "__main__":
    test_initialization(
        n_initializations=N_INITIALIZATIONS,
        n_data=N_DATA,
        loss_function=LOSS_FUNCTION,
    )
