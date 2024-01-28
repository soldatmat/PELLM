import time
import torch

from lib.model.extended.init_model import init_model
from lib.model.extended.model_states import save_model_states
from lib.model.train import train
from lib.model.progen.init_tokenizer import init_tokenizer
from lib.data.datasets.GB1 import get_GB1_dataset
from lib.utils.file import save_pt_file
from lib.data.data import resample_uniform


# Function defaults
LEARNING_RATE = 1e-3
# LOSS_FUNCTION = torch.nn.functional.mse_loss
LOSS_FUNCTION = torch.nn.functional.l1_loss
BATCH_SIZE = 1
N_EPOCHS = 1
EVALUATION_PERIOD = 1000  # [number of batches]

# Script settings
N_DATA = None  # all data = 149631
TEST_SPLIT = 0.1
TEST_DATA_INDEXES = None
# FILTER_DATA = None
# FILTER_DATA = lambda df: df.loc[df["Fitness"] >= 6.0]  # 27 variants
# FILTER_DATA = lambda df: df.iloc[[0, 1, 18, 74, 519, 623, 949, 32322, 50456, 49771]]
FILTER_DATA = lambda df: resample_uniform(df, bins=[1.0, 3.0, max(df["Fitness"])])

LOAD_MODEL = ""
SAVE_PATH = "/models"
SAVE_NAME = "data_balanced_05"

# Constants
FILE_PREPEND = "progen_extended"


def train_progen_extended(
    learning_rate=LEARNING_RATE,
    loss_function=LOSS_FUNCTION,
    n_data=None,
    filter_data=None,
    test_data_indexes=None,
    test_split=None,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    evaluation_period=EVALUATION_PERIOD,
    state_dict_path: str = None,
    absolute_paths=False,
    save_path: str = None,
    save_name: str = None,
    save_state_dict: str = None,
    save_info: str = None,
):
    if save_path and save_name:
        save_state_dict, save_info = prepare_save_paths(
            save_path, save_name, save_state_dict, save_info
        )

    print("Connecting to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Connected to {device}")

    print("Loading tokenizer")
    tokenizer = init_tokenizer()

    print("Loading GB1 data")
    tokenize = lambda sequence: tokenizer.encode(sequence).ids
    if test_data_indexes is not None:
        print(
            "Using [test_data_indexes] to select test data, [test_split] will be ignored"
        )
        test_sequences, test_fitnesses, test_variants = get_GB1_dataset(
            tokenize=tokenize,
            shuffle=False,
            data_indexes=test_data_indexes,
            device=device,
            return_variants=True,
        )
        print(f"Sampling {n_data} data for training")
        train_sequences, train_fitnesses, train_variants = get_GB1_dataset(
            tokenize=tokenize,
            filter_data=filter_data,
            shuffle=True,
            n_data=n_data,
            exclude_indexes=test_data_indexes,
            device=device,
            return_variants=True,
        )
    elif test_split is not None:
        print(f"Sampling {n_data} data with [test_split] = {test_split}")
        (
            train_sequences,
            train_fitnesses,
            test_sequences,
            test_fitnesses,
            train_variants,
            test_variants,
        ) = get_GB1_dataset(
            tokenize=tokenize,
            filter_data=filter_data,
            test_split=test_split,
            shuffle=True,
            n_data=n_data,
            device=device,
            return_variants=True,
        )
    else:
        print(f"Sampling {n_data}, using same data for training and testing")
        (
            train_sequences,
            train_fitnesses,
            train_variants,
        ) = get_GB1_dataset(
            tokenize=tokenize,
            filter_data=filter_data,
            shuffle=True,
            n_data=n_data,
            device=device,
            return_variants=True,
        )
        test_sequences = train_sequences
        test_fitnesses = train_fitnesses
        test_variants = train_variants

    print("Initializing model")
    model = init_model(
        state_dict_path=state_dict_path,
        absolute_path=absolute_paths,
    ).to(device)

    if save_info:
        info = {
            "loss_function": loss_function,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "evaluation_period": evaluation_period,
            "train_variants": train_variants,
            "test_variants": test_variants,
        }
        save_pt_file(
            info,
            save_to=save_info,
            var_name="training parameters",
        )

    print("Training")
    start_time = time.time()
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
    train_time = time.time() - start_time
    print(f"Training finished in {train_time} s")

    if save_info:
        info["train_time"] = train_time
        info["loss_history"] = loss_history
        save_pt_file(info, save_to=save_info, var_name="loss_history and train_time")

    if save_state_dict:
        save_model_states(
            model,
            save_to=save_state_dict,
            absolute_path=absolute_paths,
        )

    return loss_history


def prepare_save_paths(save_path, save_name, save_state_dict, save_info):
    if save_state_dict is None:
        save_state_dict = save_path + "/" + FILE_PREPEND + "_" + save_name + ".pt"
    if save_info is None:
        save_info = save_path + "/" + FILE_PREPEND + "_" + save_name + "_info.pt"
    return save_state_dict, save_info


if __name__ == "__main__":
    loss_history = train_progen_extended(
        state_dict_path=LOAD_MODEL,
        save_path=SAVE_PATH,
        save_name=SAVE_NAME,
        n_data=N_DATA,
        filter_data=FILTER_DATA,
        test_data_indexes=TEST_DATA_INDEXES,
        test_split=TEST_SPLIT,
    )
