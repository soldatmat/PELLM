import os
from pathlib import Path

import pandas
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import torch

GB1_PATH = "../../../../../data/GB1/elife-16965-supp1.xlsx"

WT_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
MUTATION_POSITIONS = [38, 39, 40, 53]  # [39, 40, 41, 54] - 1 for indexing


def get_GB1_dataset(
    tokenize=None,
    raw=False,
    test_split: float = None,
    shuffle=True,
    data_indexes=None,
    exclude_indexes=None,
    n_data: int = None,
    device: torch.device = None,
):
    df = load_data()
    if data_indexes:
        df = df.iloc[data_indexes]
    if exclude_indexes:
        df = df.drop(exclude_indexes)
    if shuffle:
        df = df.sample(frac=1)
    if n_data:
        df = df[:n_data]
    sequences, fitnesses = prepare_data(df)
    if raw:
        return sequences, fitnesses

    if tokenize is not None:
        sequences = tokenize_batch(sequences, tokenize)
    else:
        sequences = torch.tensor(sequences)
    fitnesses = torch.tensor(fitnesses)

    if device is not None:
        sequences = sequences.to(device)
        fitnesses = fitnesses.to(device)

    if test_split is not None:
        (
            train_sequences,
            test_sequences,
            train_fitnesses,
            test_fitnesses,
        ) = train_test_split(sequences, fitnesses, test_size=test_split, shuffle=False)
        return train_sequences, train_fitnesses, test_sequences, test_fitnesses

    return sequences, fitnesses


def load_data():
    return pandas.read_excel(os.path.dirname(Path(__file__)) + "/" + GB1_PATH)


def prepare_data(dfs: DataFrame):
    part1 = WT_SEQUENCE[: MUTATION_POSITIONS[0]]
    part2 = WT_SEQUENCE[MUTATION_POSITIONS[0] + 1 : MUTATION_POSITIONS[1]]
    part3 = WT_SEQUENCE[MUTATION_POSITIONS[1] + 1 : MUTATION_POSITIONS[2]]
    part4 = WT_SEQUENCE[MUTATION_POSITIONS[2] + 1 : MUTATION_POSITIONS[3]]
    part5 = WT_SEQUENCE[MUTATION_POSITIONS[3] + 1 :]

    sequences = []
    for variant in dfs.Variants:
        variant_sequence = (
            part1
            + variant[0]
            + part2
            + variant[1]
            + part3
            + variant[2]
            + part4
            + variant[3]
            + part5
        )
        sequences.append(variant_sequence)

    fitnesses = []
    for fitness in dfs.Fitness:
        fitnesses.append(fitness)

    return sequences, fitnesses


def tokenize_batch(sequences, tokenize):
    tokenized = torch.empty((len(sequences), len(WT_SEQUENCE)), dtype=torch.int64)
    for s in range(len(sequences)):
        tokenized[s, :] = torch.tensor(tokenize(sequences[s]))

    return tokenized
