from typing import List
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ue_abssum.default_values import (
    DEFAULT_UE_METRICS,
    DEFAULT_SEQ2SEQ_METRICS,
    REV_OUTPUT_TYPE,
)
from ue_abssum.tokenize_data import tokenize_data

def get_ats_metrics_names(ue_dict) -> List[str]:
    return list(
        set(
            (
                x.split("_")[0]
                for x in ue_dict.keys()
                if not x.startswith("ROC_AUC") and not x.startswith("data_ROC_AUС")
            )
        )
    )

def _load_model_and_tokenizer_if_necessary(generate_kwargs):
    model, tokenizer = generate_kwargs["model"], generate_kwargs["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(model, str):
        generate_kwargs["model"] = AutoModelForSeq2SeqLM.from_pretrained(model).to(
            device
        )
    if isinstance(tokenizer, str):
        generate_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer)
    return generate_kwargs


def get_default_ue_and_seq2seq_metrics_names(
    ue_metrics_names=None, seq2seq_metrics_names=None
):
    if ue_metrics_names is None:
        ue_metrics_names = DEFAULT_UE_METRICS
    if seq2seq_metrics_names is None:
        seq2seq_metrics_names = DEFAULT_SEQ2SEQ_METRICS
    return ue_metrics_names, seq2seq_metrics_names


def get_seq2seq_metrics_names(ue_dict) -> List[str]:
    return list(
        set(
            (
                x.split("_")[0]
                for x in ue_dict.keys()
                if not x.startswith("ROC_AUC") and not x.startswith("data_ROC_AUС")
            )
        )
    )


def get_ue_method_names_and_ravel_ue_dict(ue_methods, ue_dict):
    names, ue_raveled = [], []
    for name, ue_obj in zip(ue_methods, ue_dict):
        if not isinstance(ue_obj, dict):
            names.append(name)
            ue_raveled.append(ue_obj)
        else:
            for key, value in ue_obj.items():
                names.append(key)
                ue_raveled.append(value)
    return names, ue_raveled


def get_generation_max_length(train_data, tokenizer, data_config):
    tokenized_data = tokenize_data(
        train_data,
        tokenizer,
        getattr(data_config, "text_name", "document"),
        getattr(data_config, "label_name", "summary"),
    )
    gen_max_length = min(
        # Using quantile instead of `max` since `max` can lead to enormously long generation
        round(np.quantile([len(x) for x in tokenized_data["labels"]], 0.95) + 1),
        tokenizer.model_max_length,
    )
    return gen_max_length


def log_config(log, conf, num_tabs=0):
    config = to_dict(conf)
    for key, value in config.items():
        if isinstance(value, dict):
            log.info("\t" * num_tabs + key)
            log_config(log, value, num_tabs + 1)
        else:
            log.info("\t" * num_tabs + f"{key}: {value}")


def to_dict(conf):
    config = dict(conf)
    if "hydra" in config.keys():
        del config["hydra"]
    for key, value in config.items():
        if isinstance(value, DictConfig):
            config[key] = to_dict(value)
    config = dict(config)
    return config


def entropy(tensor_bn_v):
    """
    B - num bins, N - batch length, V - vocabulary size
    :param tensor_bn_v: tensor with ¡probabilities!
    """
    tensor_clipped = tensor_bn_v.clip(1e-64, 1)
    return -(tensor_clipped * tensor_clipped.log()).sum(dim=-1)


def entropy_wo_clipping(tensor_bn_v):
    return -(tensor_bn_v * tensor_bn_v.log()).sum(dim=-1)


def entropy_for_log(tensor_bn_v_log):
    """
    B - num bins, N - batch length, V - vocabulary size
    :param tensor_bn_v: tensor with ¡probabilities!
    """
    tensor_clipped = tensor_bn_v_log.clip(-100, 0)
    return -(tensor_clipped * tensor_clipped.exp()).sum(dim=-1)


def make_method_upper(method):
    splitted = method.split("+")[0]
    if "+" in method:
        return splitted.upper() + method[len(splitted) :]
    if "-" not in splitted:
        return splitted.upper()
    method_upper = ""
    parts = splitted.split("-")
    for i, part in enumerate(parts):
        if i < 2:
            method_upper += part.upper() + "-"
        else:
            method_upper += part + "-"
    return method_upper[:-1]


def _set_device(device: str) -> str:
    if torch.cuda.is_available() and device.isdigit():
        return f"cuda:{device}"
    elif torch.cuda.is_available() and (not device == "cpu"):
        return "cuda"
    return "cpu"


def get_method_name_and_strategy_kwargs(method: str, kwargs: dict, output_dict: dict):
    method_old = method
    method_name_upper = method.split("+")[0].upper()
    strategy_kwargs = {}
    generate_method = REV_OUTPUT_TYPE.get(method)
    # Check strategy kwargs
    strategy_kwargs_upd = kwargs.get(method, None) or kwargs.get(method_old, None)
    if strategy_kwargs_upd is not None:
        strategy_kwargs.update(strategy_kwargs_upd)
    if generate_method != "no":
        strategy_kwargs["inference_output"] = output_dict[generate_method]
    if method_name_upper == "BLEUVARDET":
        strategy_kwargs["stochastic_output"] = strategy_kwargs.pop("inference_output")
        strategy_kwargs["deterministic_output"] = output_dict["inference"]
    elif method.split("-")[0] in ["MD", "NUQ", "DDU", "RDE"]:
        inference_output = strategy_kwargs.pop("inference_output")
        strategy_kwargs["train_embeddings"] = inference_output["train_embeddings"]
        strategy_kwargs["test_embeddings"] = inference_output["test_embeddings"]
    elif any(
        (generate_method.lower().startswith(x) for x in ("ensemble", "ep_single"))
    ):
        # need to extract two keys
        if "SEQ" in method:
            strategy_kwargs["sequence_level_data"] = strategy_kwargs[
                "inference_output"
            ]["sequence_level_data"]
        elif "PE-TOK" in method:
            strategy_kwargs["token_level_data"] = strategy_kwargs["inference_output"][
                "pe_token_level_data"
            ]
        elif "EP-TOK" in method:
            strategy_kwargs["token_level_data"] = strategy_kwargs["inference_output"][
                "ep_token_level_data"
            ]
        else:
            raise NotImplementedError
        strategy_kwargs["inference_output"] = strategy_kwargs["inference_output"].get(
            "inference_output"
        )

    return method_name_upper, strategy_kwargs
