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
    filter_data=None,
    test_split: float = None,
    shuffle=True,
    data_indexes=None,
    exclude_indexes=None,
    n_data: int = None,
    device: torch.device = None,
    return_variants=False,
):
    def return_values():
        if test_split is None:
            if return_variants:
                return sequences, fitness, variants
            else:
                return sequences, fitness
        else:
            if return_variants:
                return (
                    train_sequences,
                    train_fitness,
                    test_sequences,
                    test_fitness,
                    train_variants,
                    test_variants,
                )
            else:
                return train_sequences, train_fitness, test_sequences, test_fitness

    df = load_data()
    if data_indexes:
        df = df.iloc[data_indexes]
    if exclude_indexes:
        df = df.drop(exclude_indexes)
    if filter_data:
        df = filter_data(df)
    if shuffle:
        df = df.sample(frac=1)
    if n_data:
        df = df[:n_data]
    fitness = df["Fitness"].values
    variants = df["Variants"].values

    if test_split is not None:
        (
            train_variants,
            test_variants,
            train_fitness,
            test_fitness,
        ) = train_test_split(variants, fitness, test_size=test_split, shuffle=False)

    if test_split is None:
        sequences = prepare_sequences(variants)
    else:
        train_sequences = prepare_sequences(train_variants)
        test_sequences = prepare_sequences(test_variants)

    if raw:
        return return_values()

    if tokenize is not None:
        if test_split is None:
            sequences = tokenize_batch(sequences, tokenize)
        else:
            train_sequences = tokenize_batch(train_sequences, tokenize)
            test_sequences = tokenize_batch(test_sequences, tokenize)
    else:
        # TODO tensor of strings is not possible
        if test_split is None:
            sequences = torch.tensor(sequences)
        else:
            train_sequences = torch.tensor(train_sequences)
            test_sequences = torch.tensor(test_sequences)

    if test_split is None:
        fitness = torch.tensor(fitness)
    else:
        train_fitness = torch.tensor(train_fitness)
        test_fitness = torch.tensor(test_fitness)

    if device is not None:
        if test_split is None:
            sequences = sequences.to(device)
            fitness = fitness.to(device)
        else:
            train_sequences = train_sequences.to(device)
            test_sequences = test_sequences.to(device)
            train_fitness = train_fitness.to(device)
            test_fitness = test_fitness.to(device)

    return return_values()


def load_data():
    return pandas.read_excel(os.path.dirname(Path(__file__)) + "/" + GB1_PATH)


def prepare_sequences(variants):
    part1 = WT_SEQUENCE[: MUTATION_POSITIONS[0]]
    part2 = WT_SEQUENCE[MUTATION_POSITIONS[0] + 1 : MUTATION_POSITIONS[1]]
    part3 = WT_SEQUENCE[MUTATION_POSITIONS[1] + 1 : MUTATION_POSITIONS[2]]
    part4 = WT_SEQUENCE[MUTATION_POSITIONS[2] + 1 : MUTATION_POSITIONS[3]]
    part5 = WT_SEQUENCE[MUTATION_POSITIONS[3] + 1 :]

    sequences = []
    for variant in variants:
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
    return sequences


def tokenize_batch(sequences, tokenize):
    """
    All [sequences] have to be the same length
    """
    tokenized = torch.empty((len(sequences), len(sequences[0])), dtype=torch.int64)
    for s in range(len(sequences)):
        tokenized[s, :] = torch.tensor(tokenize(sequences[s]))

    return tokenized
