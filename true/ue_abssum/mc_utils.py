from itertools import chain
from typing import Dict, Union, List

import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
from datasets import Dataset
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import set_seed
from copy import deepcopy

from ue_abssum.forward import forward
from ue_abssum.generate import generate


def get_mc_output(
    model,
    data,
    tokenizer,
    is_tokenized=False,
    data_config=None,
    do_sample: bool = False,
    mc_iterations: int = 10,
    generation_max_length: int = 64,
    num_return_sequences: int = 4,
    do_aug: bool = False,
    batch_size=None,
    seed=42,
    **kwargs,
) -> Dict[str, Union[List[str], np.ndarray]]:
    generate_kwargs = dict(
        model=model,
        data=data,
        tokenizer=tokenizer,
        is_tokenized=is_tokenized,
        data_config=data_config,
        generation_max_length=generation_max_length,
        num_return_sequences=num_return_sequences,
        # We do need to store all the beams from each model
        aggregate_sequences_scores=True,
        batch_size=batch_size,
        to_numpy=True,
        num_beams=kwargs.get("num_beams"),
        length_penalty=kwargs.get("length_penalty")
        if "length_penalty" in kwargs
        else 1.0,
    )

    de_models = kwargs.get("de_models", None)
    if de_models is not None:
        mc_iterations = len(de_models)
        models = [model] + [
            model.from_pretrained(checkpoint).to(model.device)
            for checkpoint in de_models
        ]

    # Check whether use beams instead as an ensemble
    if mc_iterations <= 1:
        num_beams = generate_kwargs["num_beams"]
        mc_iterations = num_beams / kwargs.get("generation_num_beams", 4)
        # TODO: probably remove
        assert (
            num_return_sequences <= num_beams
        ), f"`num_beams` must be >= `num_return_sequences`, got values {num_beams}, {num_return_sequences}"

        generate_kwargs["num_return_sequences"] = num_return_sequences
        generate_kwargs["batch_size"] = round(
            (generate_kwargs.get("batch_size", 50) or 50) / mc_iterations
        )
        if generate_kwargs.get("do_sample", False):
            generate_kwargs["batch_size"] = round(generate_kwargs["batch_size"] / 5)

    if not do_sample and not do_aug:
        # Since BART exploits F.dropout instead of nn.Dropout, we can only turn it on via .train()
        if seed is not None:
            set_seed(seed)
        model.train()
        update_kwargs = {"do_sample": False, "to_eval_mode": False}
    elif do_sample:
        # Using sampling instead of mc dropout
        model.eval()
        update_kwargs = {
            "do_sample": True,
            "to_eval_mode": True,
            "top_p": kwargs.get("generate_top_p", 0.95),
            "top_k": kwargs.get("top_k", 10),
            "early_stopping": kwargs.get("early_stopping", False),
        }
        if num_return_sequences > 1:
            # Using such a construction since batch_size may be in the dict with a value `None`
            generate_kwargs["batch_size"] = round(
                (generate_kwargs.get("batch_size", 50) or 50) / num_return_sequences
            )
    else:
        texts, text_name, generate_kwargs = _get_text_data_and_update_data_config(
            data, generate_kwargs
        )
        aug = naw.ContextualWordEmbsAug(
            model_path=kwargs.get("aug_model", "bert-base-uncased"),
            action="substitute",
            stopwords=stopwords.words("english"),
            device=model.device.type,
        )
        augmented_texts = []
        for _ in tqdm(range(mc_iterations), desc="Augmentations done..."):
            augmented_texts.append(aug.augment(texts, n=1))
        generate_kwargs["data"] = Dataset.from_pandas(
            pd.DataFrame({text_name: chain.from_iterable(augmented_texts)})
        )
        generate_kwargs["is_tokenized"] = False
        n_augs = mc_iterations
        mc_iterations = 1
        update_kwargs = {"do_sample": False, "to_eval_mode": False}

    generate_kwargs.update(update_kwargs)

    if isinstance(mc_iterations, int):
        hypotheses, scores = [], []
        for i in tqdm(range(mc_iterations), desc="MC iterations done..."):
            if de_models is not None:
                generate_kwargs["model"] = models[i]
            inference = generate(**generate_kwargs)
            scores.append(inference["scores"])  # mc_iter x num_obs
            hypotheses.append(inference["hypotheses"])  # mc_iter x num_obs
        scores = np.r_[scores].T  # num_obs x mc_iter
        hypotheses = np.r_[hypotheses].T.tolist()  # num_obs x mc_iter
    else:
        # Using beams instead as an ensemble
        inference = generate(**dict(generate_kwargs, aggregate_sequences_scores=False))
        scores = inference["scores"]  # num_obs x num_ret_seq
        hypotheses = inference["hypotheses"]  # num_obs x num_ret_seq

    # Exponentiate the scores for correct future aggregation
    scores = np.exp(scores)  # num_obs x mc_iter or num_ret_seq
    # If augmentation was used, need to reshape
    if do_aug:
        scores = scores.reshape(n_augs, -1).T
        hypotheses = np.array(hypotheses).reshape(n_augs, -1).T.tolist()
    return {"hypotheses": hypotheses, "scores": scores}


def get_mc_forward_output(
    model,
    data,
    tokenizer,
    is_tokenized=False,
    data_config=None,
    mc_iterations: int = 10,
    batch_size=None,
    seed=42,
    **kwargs,
) -> Dict[str, np.ndarray]:
    generate_kwargs = dict(
        model=model,
        data=data,
        tokenizer=tokenizer,
        is_tokenized=is_tokenized,
        data_config=data_config,
        batch_size=batch_size,
        to_numpy=True,
        to_eval_mode=False,
    )

    if seed is not None:
        set_seed(seed)
    # Since BART exploits F.dropout instead of nn.Dropout, we can only turn it on via .train()
    model.train()

    scores, losses, probas, log_probas = [], [], [], []
    for _ in tqdm(range(mc_iterations), desc="MC iterations done..."):
        inference = forward(**generate_kwargs)
        probas.append(inference["probas"])  # mc_iter x num_obs x max_seq_len
        log_probas.append(inference["log_probas"])  # mc_iter x num_obs x max_seq_len
        losses.append(inference["losses"])  # mc_iter x num_obs
        scores.append(inference["scores"])  # mc_iter x num_obs

    num_labels = inference["num_labels"]  # num_obs
    labels = inference["labels"]  # num_obs x max_seq_len

    log_probas = np.r_[log_probas].transpose(1, 0, 2)  # num_obs x mc_iter x max_seq_len
    probas = np.r_[probas].transpose(1, 0, 2)  # num_obs x mc_iter x max_seq_len
    losses = np.r_[losses].T  # num_obs x mc_iter
    scores = np.r_[scores].T  # num_obs x mc_iter

    return {
        "scores": scores,
        "losses": losses,
        "probas": probas,
        "log_probas": log_probas,
        "num_labels": num_labels,
        "labels": labels,
    }


def _get_text_data_and_update_data_config(data, generate_kwargs):
    generate_kwargs = deepcopy(generate_kwargs)
    data_config = generate_kwargs["data_config"]
    if "input_ids" in data.features:
        texts = generate_kwargs["tokenizer"].batch_decode(
            data["input_ids"], skip_special_tokens=True
        )
        generate_kwargs["data_config"]["text_name"] = "document"
    elif data_config is None or data_config.get("text_name") is None:
        texts = data["document"]
    else:
        texts = data[data_config["text_name"]]
        generate_kwargs["data_config"]["text_name"] = "document"
    text_name = "document"
    return texts, text_name, generate_kwargs
