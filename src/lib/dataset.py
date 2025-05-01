import random
from typing import Tuple

import pandas as pd


def split_training_validation(dataset: pd.DataFrame, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and validation sets.
    The dataset to split must contain a 'sentence' integer column. This function
    will ensure that the same sentence number is not present in both sets.
    Data is split in a stratified manner based on the 'sentence' column.

    The proportion of data in each set is not determined by the number of rows,
    but by the number of unique sentences. The validation set will contain a
    proportion of the unique sentences specified by val_size.

    Args:
        dataset (pd.DataFrame): The dataset to split.
        val_size (float): The proportion of the dataset sentences to include in
            the validation set. Default is 0.2 (20%).
    """
    # Ensure the dataset contains the 'sentence' column
    if "sentence" not in dataset.columns:
        raise ValueError("The dataset must contain a 'sentence' integer column.")

    unique_sentence_ids = dataset["sentence"].unique()
    random.shuffle(unique_sentence_ids)
    num_val_sentences = int(len(unique_sentence_ids) * val_size)
    val_sentence_ids = unique_sentence_ids[:num_val_sentences]
    train_sentence_ids = unique_sentence_ids[num_val_sentences:]
    train_set = dataset[dataset["sentence"].isin(train_sentence_ids)]
    val_set = dataset[dataset["sentence"].isin(val_sentence_ids)]
    return train_set, val_set
