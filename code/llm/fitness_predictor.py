import time
import torch
from sklearn.model_selection import train_test_split

from lib.model.train import train
from lib.model.single_layer.SingleLayer import SingleLayer
from lib.model.single_layer.TwoLayer import TwoLayer
from lib.utils.file import save_pt_file
from lib.utils.path import prepare_save_paths
from lib.data.data import resample_uniform
from lib.model.extended.model_states import save_model_states
from lib.functional.wieghted_l1_loss import weighted_l1_Loss


# DATA_PATH = "./../../data/GB1/progen2_dataframe.pt"
DATA_PATH = "./../../data/GB1/esm-1b_dataframe.pt"

# INPUT_PATH = "./../../data/GB1/progen2_embedding.pt"
# FITNESS_PATH = "./../../data/GB1/progen2_fitness.pt"
# LABELS_PATH = "./../../data/GB1/progen2_variants.pt"

TEST_SPLIT = None

LOSS_FUNCTION = torch.nn.functional.l1_loss
# LOSS_FUNCTION = weighted_l1_Loss()

# Tested with resample_uniform(df, bins=[1.0, max(df["Fitness"])])
# LEARNING_RATE = 1e-7 # Signmoid Identity ProGen2 1l
LEARNING_RATE = 1e-4  # Sigmoid ESM-1b 2l
# LEARNING_RATE = 1e-5  # Sigmoid ProGen2 2l

NORMALIZED_FITNESS = True

BATCH_SIZE = 1
N_EPOCHS = 5
EVALUATION_PERIOD = 10

# FILTER_DATA = None
FILTER_DATA = lambda df: resample_uniform(df, bins=[6.0, max(df["Fitness"])])
# FILTER_DATA = lambda df: resample_uniform(df, bins=[0.0, 0.5])[0:1000]

# EMBEDDING_SIZE = 1024  # Progen2
EMBEDDING_SIZE = 1280  # ESM-1b

# MODEL = SingleLayer(activation_function=torch.nn.Sigmoid(), embedding_size=EMBEDDING_SIZE)
# MODEL = SingleLayer(activation_function=torch.nn.LeakyReLU(), embedding_size=EMBEDDING_SIZE)
# MODEL = SingleLayer(activation_function=torch.nn.Identity(), embedding_size=EMBEDDING_SIZE)
MODEL = TwoLayer(activation_function=torch.nn.Sigmoid(), embedding_size=EMBEDDING_SIZE)
# MODEL = TwoLayer(activation_function=torch.nn.Identity(), embedding_size=EMBEDDING_SIZE)

SAVE_PATH = "/models/two_layer_esm-1b"
SAVE_NAME = "sigmoid_05"
# SAVE_NAME = "leakyRELU_01"
# SAVE_NAME = "identity_01"

# Constants
FILE_PREPEND = "two_layer_esm-1b"


def train_predictor(
    model,
    data_path,
    test_split,
    loss_function,
    batch_size,
    learning_rate,
    n_epochs,
    evaluation_period,
    save_path,
    save_name,
    save_info=True,
    filter_data=None,
    normalized_fitness=True,
):
    """
    [data_path]:    relative path to a ".pt" file with pandas DataFrame
                    with columns 'Variants', 'Embedding', 'Fitness' (or 'Fitness_norm' if `normalized_fitness` is True)
    """
    save_state_dict, save_info = prepare_save_paths(
        save_path,
        FILE_PREPEND,
        save_name,
        save_state_dict=None,
        save_info=None,
    )
    print("Connecting to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Connected to {device}")

    print("Initializing model")
    model.to(device)

    print("Loading data")
    data = torch.load(data_path)

    if filter_data is not None:
        print("Filtering data")
        data = filter_data(data)

    if test_split is not None:
        (
            train_data,
            test_data,
        ) = train_test_split(data, test_size=test_split, shuffle=True)
    else:
        print("Using the same data for training and evaluation")
        train_data = data
        test_data = data

    train_embedding = torch.stack(tuple(train_data.Embedding.values)).to(device)
    test_embedding = torch.stack(tuple(test_data.Embedding.values)).to(device)

    if normalized_fitness:
        print("Using fitness values normalized to <0,1>")
        train_fitness = torch.tensor(train_data.Fitness_norm.values).to(device)
        test_fitness = torch.tensor(test_data.Fitness_norm.values).to(device)
    else:
        train_fitness = torch.tensor(train_data.Fitness.values).to(device)
        test_fitness = torch.tensor(test_data.Fitness.values).to(device)

    if save_info:
        info = {
            "loss_function": loss_function,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "evaluation_period": evaluation_period,
            "test_split": test_split,
            "train_variants": train_data.Variants.values,
            "test_variants": test_data.Variants.values,
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
        train_data=train_embedding,
        train_labels=train_fitness,
        test_data=test_embedding,
        test_labels=test_fitness,
        loss_function=loss_function,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        evaluation_period=evaluation_period,
        device=device,
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
            absolute_path=False,
        )

    return loss_history


if __name__ == "__main__":
    loss_history = train_predictor(
        model=MODEL,
        data_path=DATA_PATH,
        test_split=TEST_SPLIT,
        loss_function=LOSS_FUNCTION,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        evaluation_period=EVALUATION_PERIOD,
        save_path=SAVE_PATH,
        save_name=SAVE_NAME,
        save_info=True,
        filter_data=FILTER_DATA,
        normalized_fitness=NORMALIZED_FITNESS,
    )
