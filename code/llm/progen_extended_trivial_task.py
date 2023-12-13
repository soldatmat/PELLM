import random
import torch
from sklearn.model_selection import train_test_split

from lib.model.extended.init_model import init_model
from lib.model.extended.model_states import save_model_states
from lib.model.train import train
from lib.model.progen.init_tokenizer import init_tokenizer
from lib.data.datasets.GB1 import get_GB1_dataset, tokenize_batch
from lib.utils.file import save_pt_file


# Function defaults
LEARNING_RATE = 1e-3
# LOSS_FUNCTION = torch.nn.functional.mse_loss
LOSS_FUNCTION = torch.nn.functional.l1_loss
BATCH_SIZE = 100
N_EPOCHS = 1
EVALUATION_PERIOD = 100  # [number of batches]

# Script settings
EXTRACT_LABELS = lambda sequences : extract_number_of('A', sequences)
N_DATA = None  # all data = 149631
TEST_SPLIT = 0.01
FILTER_DATA = None

LOAD_MODEL = ""
SAVE_PATH = "/models"
SAVE_NAME = "trivial_A_02"

# Constants
FILE_PREPEND = "progen_extended"


def train_progen_extended(
    extract_labels,
    learning_rate=LEARNING_RATE,
    loss_function=LOSS_FUNCTION,
    n_data=None,
    filter_data=None,
    test_split=None,
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
    print(f"Connected to {device}")

    print("Loading tokenizer")
    tokenizer = init_tokenizer()

    #print(f"Loading {n_data} raw GB1 data")
    #(
    #    sequences,
    #    fitnesses,
    #) = get_GB1_dataset(
    #    filter_data=filter_data,
    #    shuffle=True,
    #    n_data=n_data,
    #    raw=True
    #)

    print(f"Generating data")
    sequences = generate_sequences(100000, 100)

    print(f"Extracting data labels")
    labels = extract_labels(sequences).to(device)

    print("Tokenizing sequences")
    tokenize = lambda sequence: tokenizer.encode(sequence).ids
    sequences = tokenize_batch(sequences, tokenize).to(device)

    print(f"Splitting data with [test_split] = {test_split}")
    (
        train_sequences,
        test_sequences,
        train_labels,
        test_labels,
    ) = train_test_split(sequences, labels, test_size=test_split, shuffle=False)

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
        train_labels=train_labels,
        test_data=test_sequences,
        test_labels=test_labels,
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


def extract_number_of(symbol, sequences):
    counts = torch.zeros(len(sequences), dtype=torch.int32)
    for s in range(len(sequences)):
        counts[s] = sequences[s].count(symbol)
    return counts


def generate_sequences(n_sequences, sequence_length):
    sequences = []
    for s in range(n_sequences):
        number_of_A = random.randint(0, sequence_length)
        alphabet = number_of_A*'A'+(sequence_length-number_of_A)*'C'
        sequences.append(''.join(random.choice(alphabet) for i in range(sequence_length)))
    return sequences


if __name__ == "__main__":
    loss_history = train_progen_extended(
        extract_labels=EXTRACT_LABELS,
        state_dict_path=LOAD_MODEL,
        save_path=SAVE_PATH,
        save_name=SAVE_NAME,
        n_data=N_DATA,
        filter_data=FILTER_DATA,
        test_split=TEST_SPLIT,
    )
