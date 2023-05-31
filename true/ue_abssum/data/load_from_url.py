import gzip
import json
import logging
import os

import numpy as np
import pandas as pd
import pytreebank
import wget
from datasets import Dataset, DatasetDict
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .preprocessing import (
    _use_test_subset,
    _use_train_subset,
)

log = logging.getLogger(__name__)


def load_data_from_url(config, cache_dir):
    LOAD_FUNCS_AND_ARGS = {
        "amazon": (load_amazon_5core, [config, cache_dir]),
        "20newsgroups": (load_20newsgroups, config),
        "sst5": (load_sst5, config),
        "wmt20": (load_wmt20, config),
    }
    load_func, args = LOAD_FUNCS_AND_ARGS[config.dataset_name]
    return load_func(args)


def load_amazon_5core(config, cache_dir=None):
    """Return closest version of Amazon Reviews Sports & Outdoors split from the paper
    Towards More Accurate Uncertainty Estimation In Text Classification.
    """
    texts, targets = [], []
    # get zipped dataset
    url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz"
    save_path = os.path.join(cache_dir, "amazon_5core.json.gz")
    # check if file already exists, load if not
    if not (os.path.isfile(save_path)):
        save_path = wget.download(url, out=save_path)
    # unzip it and extract data to arrays
    with gzip.open(save_path, "rb") as f:
        for line in f.readlines():
            data = json.loads(line)
            texts.append(data["reviewText"])
            targets.append(np.int64(data["overall"]))
    # to shift classes from 1-5 to 0-4
    targets = np.asarray(targets) - 1
    # split on train|val|test
    seed = getattr(config, "seed", 42)
    text_buf, text_eval, targ_buf, targ_eval = train_test_split(
        texts, targets, test_size=0.1, random_state=seed
    )
    text_train, text_val, targ_train, targ_val = train_test_split(
        text_buf, targ_buf, test_size=2.0 / 9.0, random_state=seed
    )
    amazon_train = {"label": targ_train, "text": text_train}
    amazon_eval = {"label": targ_eval, "text": text_eval}
    train_dataset = Dataset.from_dict(amazon_train)
    dev_dataset = test_dataset = Dataset.from_dict(amazon_eval)
    return train_dataset, dev_dataset, test_dataset


def load_20newsgroups(config):
    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_train = {
        "label": newsgroups_train["target"],
        "text": newsgroups_train["data"],
    }
    newsgroups_eval = fetch_20newsgroups(subset="test")
    newsgroups_eval = {
        "label": newsgroups_eval["target"],
        "text": newsgroups_eval["data"],
    }
    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(newsgroups_train),
            "validation": Dataset.from_dict(newsgroups_eval),
        }
    )
    train_dataset = Dataset.from_dict(newsgroups_train)
    dev_dataset = test_dataset = Dataset.from_dict(newsgroups_eval)
    return train_dataset, dev_dataset, test_dataset


def load_sst5(config):
    dataset = pytreebank.load_sst()
    sst_datasets = {}
    for category in ["train", "test", "dev"]:
        df = {"text": [], "label": []}
        for item in dataset[category]:
            df["text"].append(item.to_labeled_lines()[0][1])
            df["label"].append(item.to_labeled_lines()[0][0])
        cat_name = category if category != "dev" else "validation"
        sst_datasets[cat_name] = Dataset.from_dict(df)
    dataset = DatasetDict(sst_datasets)
    train_dataset = sst_datasets["train"]
    dev_dataset = sst_datasets[cat_name]
    test_dataset = sst_datasets["test"]
    return train_dataset, dev_dataset, test_dataset


def load_wmt20(config):
    def read_data_from_files(data_path, filenames, keys):
        data = {}
        max_size = 10e6
        for key in keys:
            key_text = []
            for filename in filenames:
                with open(f"{data_path}/{filename}.{key}", "r") as txt_file:
                    for i, line in tqdm(enumerate(txt_file)):
                        key_text.append(line)
                        if i >= (max_size - 1):
                            break
            data[key] = key_text
        return data

    languages = [config.text_name, config.label_name]
    data_path = (
        config.data_path
        if "data_path" in config.keys()
        else "/home/artem.vazhentsev/projects/ya_shifts/wmt20_en_ru/tmp"
    )
    dev_data_path = (
        config.eval_data_path
        if "eval_data_path" in config.keys()
        else "/home/artem.vazhentsev/projects/ya_shifts"
    )
    eval_data_path = (
        config.eval_data_path
        if "eval_data_path" in config.keys()
        else "/home/artem.vazhentsev/projects/ya_shifts/orig/eval-data"
    )

    train_filenames = (
        [config.train_filename] if "train_filename" in config.keys() else ["train"]
    )
    dev_filenames = ["orig/dev-data/reddit_dev", "wmt20_en_ru/tmp/test19"]
    eval_filenames = ["reddit_eval", "global_voices_eval"]

    train_data = read_data_from_files(data_path, train_filenames, languages)
    dev_data = read_data_from_files(dev_data_path, dev_filenames, languages)
    eval_data = read_data_from_files(eval_data_path, eval_filenames, languages)

    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)

    eval_data_df = pd.DataFrame(eval_data)
    eval_data_df = eval_data_df[eval_data_df.ru.str.len() > 3]
    test_dataset = Dataset.from_pandas(eval_data_df, preserve_index=False)

    if getattr(config, "use_subset", None) is not None:
        train_dataset = _use_train_subset(
            train_dataset,
            config.use_subset,
            getattr(config, "seed", 42),
            config.label_name,
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

    log.info(f"Loaded train size: {len(train_dataset)}")
    log.info(f"Loaded dev size: {len(dev_dataset)}")
    log.info(f"Loaded test size: {len(test_dataset)}")

    return train_dataset, dev_dataset, test_dataset


# def load_twitter_hso(config):
#
#     dataset = load_dataset('hate_speech_offensive', cache_dir=config.cache_dir)
#     df = dataset['train'].to_pandas()
#     annotators_count_cols = ['hate_speech_count', 'offensive_language_count', 'neither_count']
#
#     #split by ambiguity (for test select most ambiguous part by annotators disagreement)
#     df_test = df[df['count'] != df[annotators_count_cols].max(axis=1)].reset_index(drop=True)
#     df_train = df[df['count'] == df[annotators_count_cols].max(axis=1)].reset_index(drop=True)
#
#     train_dataset = {'label': df_train['class'],
#                      'text': df_train['tweet']}
#
#     eval_dataset = {'label': df_test['class'],
#                     'text': df_test['tweet']}
#
#     datasets = DatasetDict({'train': Dataset.from_dict(train_dataset),
#                             'validation': Dataset.from_dict(eval_dataset)})
#
#     return datasets
