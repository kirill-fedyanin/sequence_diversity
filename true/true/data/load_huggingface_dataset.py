import logging
from pathlib import Path
import datetime

import numpy as np
from datasets import load_dataset, Dataset
from pandas import DataFrame

from .preprocessing import (
    _add_id_column_to_datasets,
    _use_test_subset,
    _use_train_subset,
    _filter_quantiles,
    _multiply_data,
)

log = logging.getLogger()


class HuggingFaceDatasetsReader:
    def __init__(
        self, *dataset_args, cache_dir=None, extra_key=None, use_auth_token=None, additional_dataset_kwargs={}
    ):
        cache_dir = cache_dir / datetime.datetime.now().strftime('%H:%M:%S%f_%m.%d.%Y')
        print(cache_dir)
        self.dataset = load_dataset(
            *dataset_args, cache_dir=cache_dir, download_mode='force_redownload', use_auth_token=use_auth_token, **additional_dataset_kwargs
        )
        self.extra_key = extra_key

    def __call__(self, phase, text_name=None, label_name=None):
        dataset = self.dataset[phase]
        if self.extra_key is not None:
            dataset = Dataset.from_pandas(DataFrame(dataset[self.extra_key][:1000000]))
        if text_name is not None and label_name is not None:
            dataset = dataset.remove_columns(
                [
                    x
                    for x in dataset.column_names
                    if x not in [text_name, label_name, "id"]
                ]
            )
        setattr(self, phase, dataset)
        return getattr(self, phase)


def load_huggingface_dataset(config, cache_dir=None):
    data_cache_dir = Path(cache_dir) / "data" if cache_dir is not None else None
    text_name = config.text_name
    label_name = config.label_name

    extra_key = config.get("extra_key", None)
    dataset_params = config.get("dataset_params", {})

    kwargs = {
        "cache_dir": data_cache_dir,
        "use_auth_token": config.get("use_auth_token", None),
        "extra_key": extra_key,
        "additional_dataset_kwargs": dataset_params
    }

    hfdreader = (
        HuggingFaceDatasetsReader(config.dataset_name, **kwargs)
        if isinstance(config.dataset_name, str)
        else HuggingFaceDatasetsReader(*list(config.dataset_name), **kwargs)
    )

    if config.get("multiply_data", None) is not None:
        hfdreader = _multiply_data(hfdreader, config.multiply_data)

    train_dataset = hfdreader("train", text_name, label_name)
    dev_dataset = hfdreader("validation", text_name, label_name)
    test_dataset = hfdreader("test", text_name, label_name)

    log.info(f"Loaded train size: {len(train_dataset)}")
    log.info(f"Loaded dev size: {len(dev_dataset)}")
    log.info(f"Loaded test size: {len(test_dataset)}")

    if getattr(config, "filter_quantiles", None) is not None:
        train_dataset = _filter_quantiles(
            train_dataset,
            config.filter_quantiles,
            cache_dir,
            text_name,
            config.tokenizer_name,
        )

    if getattr(config, "use_subset", None) is not None:
        train_dataset = _use_train_subset(
            train_dataset,
            config.use_subset,
            getattr(config, "seed", 42),
            label_name,
        )
        log.info(f"Subsampled train size: {len(train_dataset)}")

    if ("id" not in train_dataset.column_names) and config.get("add_id_column", True):
        train_dataset, dev_dataset, test_dataset = _add_id_column_to_datasets(
            [train_dataset, dev_dataset, test_dataset]
        )

    if getattr(config, "use_test_subset", False):
        test_dataset, subsample_idx = _use_test_subset(
            test_dataset,
            config.use_test_subset,
            getattr(config, "seed", 42),
            getattr(config, "subset_fixed_seed", False),
        )
        dev_dataset = dev_dataset.select(
            np.setdiff1d(np.arange(len(dev_dataset)), subsample_idx)
        )
        log.info(f"Subsampled dev size: {len(dev_dataset)}")
        log.info(f"Subsampled test size: {len(test_dataset)}")

    return [train_dataset, dev_dataset, test_dataset]
