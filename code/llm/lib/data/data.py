from datasets import Dataset
import pandas
import numpy as np


def create_dataset(data, labels):
    dataset = Dataset.from_list(data)
    dataset = dataset.add_column("labels", labels)
    return dataset


def resample_uniform(df, bins):
    """
    df      pandas.dataframe, required column: "Fitness"
    bins    is passed to numpy.histogram, can be any shape that numpy.histogram accepts
    """
    counts, bins = np.histogram(
        df["Fitness"].values, bins=bins
    )

    sampled_bins = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            # last bin is inclusive on both sides
            df_bin = df.loc[(df["Fitness"] >= bins[i]) & (df["Fitness"] <= bins[i + 1])]
        else:
            df_bin = df.loc[(df["Fitness"] >= bins[i]) & (df["Fitness"] < bins[i + 1])]
        sampled_bins.append(
            df_bin.iloc[
                np.random.choice(
                    np.arange(0, len(df_bin)), np.amin(counts), replace=False
                )
            ]
        )
    return pandas.concat(sampled_bins)
