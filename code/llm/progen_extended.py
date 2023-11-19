import torch

from lib.model.progen.extended.init_model import init_model
from lib.model.progen.extended.model_states import save_model_states
from lib.model.train import train
from lib.model.progen.init_tokenizer import init_tokenizer
from lib.data.datasets.GB1 import get_GB1_dataset
from lib.utils.file import save_pt_file


# Function defaults
LEARNING_RATE = 1e-3
LOSS_FUNCTION = torch.nn.functional.mse_loss
N_DATA = 1100
TEST_SPLIT = 100 / N_DATA  # all data = 149631
BATCH_SIZE = 1
N_EPOCHS = 1
EVALUATION_PERIOD = 1 # in number of batches

# Script settings
LOAD_MODEL = ""
SAVE_PATH = "/models"
SAVE_NAME = "v1_01"

# Constants
FILE_PREPEND = "progen_extended"


def train_progen_extended(
    learning_rate=LEARNING_RATE,
    loss_function=LOSS_FUNCTION,
    n_data=N_DATA,
    test_split=TEST_SPLIT,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    evaluation_period=EVALUATION_PERIOD,

    state_dict_path: str = None,
    absolute_paths=False,

    save_path: str = None,
    save_name: str = None,
    save_state_dict: str = None,
    save_history: str = None,
    save_train_params: str = None,
):
    if save_path and save_name:
        save_state_dict, save_history, save_train_params = prepare_save_paths(
            save_path, save_name, save_state_dict, save_history, save_train_params
        )

    print("Connecting to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer")
    tokenizer = init_tokenizer()

    print("Loading GB1 data")
    tokenize = lambda sequence: tokenizer.encode(sequence).ids
    train_sequences, train_fitnesses, test_sequences, test_fitnesses = get_GB1_dataset(
        tokenize=tokenize,
        test_split=test_split,
        shuffle=True,
        n_data=n_data,
        device=device,
    )

    print("Initializing model")
    model = init_model(
        state_dict_path=state_dict_path,
        absolute_path=absolute_paths,
    ).to(device)

    print("Training")
    loss_history = train(
        model=model,
        device=device,
        train_data=train_sequences,
        train_labels=train_fitnesses,
        test_data=test_sequences,
        test_labels=test_fitnesses,
        loss_function=loss_function,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        evaluation_period=evaluation_period,
    )

    if save_state_dict:
        save_model_states(
            model,
            save_to=save_state_dict,
            absolute_path=absolute_paths,
        )

    if save_history:
        save_pt_file(loss_history, save_to=save_history, var_name="loss_history")
        save_pt_file(
            {
                "loss_function": loss_function,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "evaluation_period": evaluation_period,
            },
            save_to=save_train_params,
            var_name="training parameters",
        )

    return loss_history


def prepare_save_paths(
    save_path, save_name, save_state_dict, save_history, save_train_params
):
    save_state_dict = save_path + "/" + FILE_PREPEND + "_" + save_name + ".pt"
    save_history = save_path + "/" + FILE_PREPEND + "_" + save_name + "_history.pt"
    save_train_params = (
        save_path + "/" + FILE_PREPEND + "_" + save_name + "_train_params.pt"
    )
    return save_state_dict, save_history, save_train_params


if __name__ == "__main__":
    loss_history = train_progen_extended(
        state_dict_path=LOAD_MODEL,
        save_path=SAVE_PATH,
        save_name=SAVE_NAME,
    )
