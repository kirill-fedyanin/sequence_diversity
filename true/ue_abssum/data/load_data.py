from typing import Tuple

from datasets import Dataset

from .load_from_json_or_csv import load_from_json_or_csv
from .load_from_url import load_data_from_url
from .load_huggingface_dataset import load_huggingface_dataset


def load_data(config, cache_dir=None) -> Tuple[Dataset, Dataset, Dataset]:
    """

    :param config:
    :param cache_dir:
    :return: train_dataset, dev_dataset, test_dataset
    """
    if config.path == "url":
        return load_data_from_url(config, cache_dir)
    elif config.path != "datasets":
        return load_from_json_or_csv(config, cache_dir)
    return load_huggingface_dataset(config, cache_dir)
