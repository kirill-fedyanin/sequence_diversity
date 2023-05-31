import gc
from typing import Union
from copy import copy

import torch
import pickle
from datasets import Dataset
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer

from true.default_values import DEFAULT_TOKEN_LEVEL_MEASURES, TOP_K
from true.ensemble_generator import EnsembleGenerator
from true.generate_utils import (restore_token_level_data,
                                 update_token_level_scores,
                                 from_text_to_id,
                                 collect_token_level_uncertainties,
                                 get_collect_fn)
from true.tokenize_data import tokenize_data
from true.model_modifications import _get_dim
from true.get_embeddings import get_embeddings_from_output


def generate(
    model: AutoModelForSeq2SeqLM,
    data: Dataset,
    tokenizer: AutoTokenizer,
    is_tokenized=False,
    data_config=None,
    to_numpy: bool = False,
    to_eval_mode: bool = True,
    generation_max_length: int = None,
    num_return_sequences: int = 1,
    aggregate_sequences_scores: bool = False,
    batch_size: int = None,
    t_for_weights: Union[float, int] = 1.0,
    kg_id_mapping: dict = None,
    **kwargs,
):
    if not is_tokenized:
        if data_config is None:
            data_config = {
                "text_name": "document",
                "label_name": "summary",
            }
        data = tokenize_data(
            tokenizer=tokenizer,
            data=data,
            document_name=data_config["text_name"],
            label_name=data_config["label_name"],
        )
        if "labels" in data[0]:
            data = data.remove_columns(["labels"])

    using_ensemble = kwargs.pop("using_ensemble", None) or isinstance(
        model, EnsembleGenerator
    )
    embed_kwargs = kwargs.pop("embed_kwargs", None)

    output = _model_generate_loop(
        model=model,
        data=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        to_eval_mode=to_eval_mode,
        generation_max_length=generation_max_length,
        num_return_sequences=num_return_sequences,
        using_ensemble=using_ensemble,
        embed_kwargs=embed_kwargs,
        **kwargs,
    )

    if embed_kwargs is None:
        del output["embeddings"]

    hypotheses = tokenizer.batch_decode(
        output["sequences"].cpu(), skip_special_tokens=True
    )
    output["hypotheses"] = [
        hypotheses[i * num_return_sequences : (i + 1) * num_return_sequences]
        for i in range(len(hypotheses) // num_return_sequences)
    ]
    output["sequences"] = output["sequences"].reshape(
        -1, num_return_sequences, output["sequences"].shape[1]
    )

    for key in ["scores", "log_scores", "scores_unbiased", "entropy"]:
        output[key] = output[key].reshape(-1, num_return_sequences)
    output["max_scores"] = output["scores"].max(-1).values

    if kg_id_mapping is not None:
        batch_wd_ids = []
        for batch_hypos in output["hypotheses"]:
            wd_ids = []
            for hypo in batch_hypos:
                wd_ids.append(from_text_to_id(kg_id_mapping, hypo))
            batch_wd_ids.append(wd_ids)

        output["wd_ids"] = batch_wd_ids

    if aggregate_sequences_scores:
        # Save original data
        output["full_scores"] = output["scores"]
        # Otherwise we already have the weights
        if not using_ensemble:
            # Calculate weights
            scores_in_degree = output["scores"] ** t_for_weights
            output["weights"] = scores_in_degree / scores_in_degree.sum(
                -1, keepdims=True
            )
        else:
            output["weights"] = torch.Tensor(output["weights"]).to(output["log_scores"])
        # Modify
        output["hypotheses"] = [x[0] for x in output["hypotheses"]]
        output["sequences_wo_agg"] = output["sequences"]
        output["sequences"] = torch.stack([x[0] for x in output["sequences"]])
        for key in ["scores", "log_scores", "scores_unbiased", "entropy",
                    "entropy_top5", "entropy_top10", "entropy_top15"]:
            output[key] = (output[key] * output["weights"]).sum(-1)

    if to_numpy:
        output = {
            k: (
                v.cpu().detach().numpy()
                if isinstance(v, torch.Tensor)
                and k
                not in [
                    "wd_ids",
                    "hypotheses",
                    "max_seq_length",
                    "model_config",
                    "token_level_scores",
                ]
                else v
            )
            for k, v in output.items()
        }

    torch.cuda.empty_cache()
    gc.collect()

    return output


def _model_generate_loop(
    model: AutoModelForSeq2SeqLM,
    data: Dataset,
    tokenizer: AutoTokenizer,
    batch_size=None,
    to_eval_mode=True,
    dropout_rates = None,
    inference_seed = None,
    generation_max_length: int = None,
    num_return_sequences: int = 1,
    using_ensemble: bool = False,
    embed_kwargs: dict = None,
    **kwargs,
):
    """
    Implemented for seq2seq tasks
    Args:
        data:

    Returns: dict with sequences of ids and their scores

    """
    torch.cuda.empty_cache()
    if batch_size is None:
        batch_size = 50
    if "labels" in data.features:
        data = data.remove_columns(["labels"])

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest"),
    )

    if generation_max_length is not None:
        max_length = sequences_length = generation_max_length
    else:
        max_length = None
        sequences_length = 128

    if to_eval_mode:
        model.eval()
    device = model.device
    sequences_X_shape = num_return_sequences * len(data)
    sequences = (
        torch.zeros(
            (sequences_X_shape, sequences_length), dtype=torch.int64, device=device
        )
        + tokenizer.pad_token_id
    )
    log_scores = torch.empty(sequences_X_shape, dtype=torch.float32, device=device)
    max_seq_len = 0

    model_config = model.config
    if "mbart" in model_config._name_or_path:
        model_config.decoder_start_token_id = tokenizer.lang_code_to_id[
            tokenizer.tgt_lang
        ]
        kwargs["decoder_start_token_id"] = model_config.decoder_start_token_id

    base_keys = ["entropy", "entropy_s", "entropy_s_u", "scores_unbiased", "beam_weights", "weights"]
    base_keys = {
        key: None
        for key in base_keys + [f"entropy_top{k}" for k in TOP_K]
    }

    if using_ensemble:
        base_keys = base_keys.update({"probas": None, "log_probas": None})
        ep_token_level_scores = copy(base_keys)
        pe_token_level_scores = copy(base_keys) 

        ep_token_level_scores.update({key: None for key in DEFAULT_TOKEN_LEVEL_MEASURES})
        pe_token_level_scores.update({key: None for key in DEFAULT_TOKEN_LEVEL_MEASURES})

        ensembling_mode = kwargs.get("ensembling_mode", "pe")
    else:
        token_level_scores = base_keys

    if kwargs.get("num_beams") is None and ('do_sample' not in kwargs or kwargs['do_sample'] == None):
        kwargs["num_beams"] = num_return_sequences

    if embed_kwargs is not None:
        num_obs = len(data)
        dim = _get_dim(model)
        embeddings = torch.empty((num_obs, dim), dtype=torch.float, device=device)
        embeddings_decoder = torch.empty(
            (num_obs, dim), dtype=torch.float, device=device
        )
        hidden_state = embed_kwargs.get("hidden_state", "encoder")
        kwargs["output_hidden_states"] = True

    seed = kwargs.get("inference_seed", None)
    if seed is not None:
        del kwargs['inference_seed']
    models_scores = []
    with torch.no_grad():
        start = 0
        start_embed = 0
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if seed is not None:
                torch.manual_seed(seed)

            output = model.generate(
                **batch,
                max_length=max_length,
                min_length=3,  # To avoid empty hypotheses. 3 == <BOS> + at least one token + <EOS>
                output_scores=True,
                return_dict_in_generate=True,
                num_return_sequences=num_return_sequences,
                **kwargs,
            )

            batch_length = len(batch["input_ids"])
            end = start + (batch_length * num_return_sequences)
            end_embed = start_embed + batch_length
            if embed_kwargs is not None:
                batch_embeddings, batch_embeddings_decoder = get_embeddings_from_output(
                    output,
                    None,
                    batch,
                    hidden_state=hidden_state,
                    ignore_padding=embed_kwargs.get("ignore_padding", True),
                    extraction_method="generate",
                    use_averaging=embed_kwargs.get("use_averaging", True),
                    all_layers=embed_kwargs.get("all_layers", False),
                    aggregation_method=embed_kwargs.get("aggregation_method", "mean")
                )
                if "encoder" in hidden_state:
                    embeddings[start_embed:end_embed].copy_(
                        batch_embeddings, non_blocking=True
                    )
                if "decoder" in hidden_state:
                    embeddings_decoder[start_embed:end_embed].copy_(
                        batch_embeddings_decoder, non_blocking=True
                    )

            batch_length = len(batch["input_ids"])
            end = start + (batch_length * num_return_sequences)
            sequences[start:end, : output.sequences.shape[1]].copy_(
                output.sequences, non_blocking=True
            )
            collect_fn = get_collect_fn(output)
            collect_fn = partial(collect_fn,
                                 output,
                                 batch_size,
                                 num_return_sequences,
                                 model.config.vocab_size,
                                 model.config.pad_token_id)
            if using_ensemble:
                pe_token_level_uncertainties = \
                    collect_fn(
                        ensemble_uncertainties=output['pe_uncertainties']
                    )
                ep_token_level_uncertainties = \
                    collect_fn(
                        ensemble_uncertainties=output['ep_uncertainties']
                    )

                ep_token_level_scores = update_token_level_scores(
                    ep_token_level_scores, ep_token_level_uncertainties
                )
                pe_token_level_scores = update_token_level_scores(
                    pe_token_level_scores, pe_token_level_uncertainties
                )
            else:
                batch_tok_lev_scores = collect_fn()
                token_level_scores = update_token_level_scores(
                    token_level_scores, batch_tok_lev_scores
                )
                if 'sequences_scores' in output:
                    log_scores[start:end].copy_(output.sequences_scores, non_blocking=True)
                else:
                    log_scores[start:end].copy_(batch_tok_lev_scores['sequences_scores'], non_blocking=True)
            start = end
            start_embed = end_embed
            max_seq_len = max(max_seq_len, output.sequences.shape[1])

    embeddings_dict = {}
    if embed_kwargs is not None:
        if "encoder" in hidden_state:
            embeddings_dict["encoder"] = embeddings
        if "decoder" in hidden_state:
            embeddings_dict["decoder"] = embeddings_decoder
    # TODO: rename `scores` to `probas`
    output_dict = {
        "sequences": sequences[:, :max_seq_len],
        "scores": log_scores.exp(),
        "log_scores": log_scores,
        "max_seq_length": max_seq_len,
        "model_config": model_config,
        "embeddings": embeddings_dict
    }
    
    if using_ensemble:
        output_dict.update({
            "pe_token_level_scores": pe_token_level_scores,
            "ep_token_level_scores": ep_token_level_scores,
            "probas": pe_token_level_scores["probas"],
            "log_probas": pe_token_level_scores["log_probas"]
        })
        if ensembling_mode == 'pe':
            output_dict.update({
                "weights": torch.Tensor(pe_token_level_scores["weights"]).to(log_scores),
                "scores_unbiased": torch.Tensor(pe_token_level_scores["scores_unbiased"]).to(
                    log_scores
                ),
                "entropy": torch.Tensor(pe_token_level_scores["entropy"]).to(log_scores),
                "entropy_top5": torch.Tensor(pe_token_level_scores["entropy_top5"]).to(log_scores),
                "entropy_top10": torch.Tensor(pe_token_level_scores["entropy_top10"]).to(log_scores),
                "entropy_top15": torch.Tensor(pe_token_level_scores["entropy_top15"]).to(log_scores),
            })
        elif ensembling_mode == 'ep':
            output_dict.update({
                "weights": torch.Tensor(ep_token_level_scores["weights"]).to(log_scores),
                "scores_unbiased": torch.Tensor(ep_token_level_scores["scores_unbiased"]).to(
                    log_scores
                ),
                "entropy": torch.Tensor(ep_token_level_scores["entropy"]).to(log_scores),
                "entropy_top5": torch.Tensor(ep_token_level_scores["entropy_top5"]).to(log_scores),
                "entropy_top10": torch.Tensor(ep_token_level_scores["entropy_top10"]).to(log_scores),
                "entropy_top15": torch.Tensor(ep_token_level_scores["entropy_top15"]).to(log_scores),
            })
        else:
            raise NotImplementedError
    else:
        output_dict.update({
            "scores_unbiased": torch.Tensor(token_level_scores["scores_unbiased"]).to(
                log_scores
            ),
            "entropy": torch.Tensor(token_level_scores["entropy"]).to(log_scores),
            "entropy_top5": torch.Tensor(token_level_scores["entropy_top5"]).to(log_scores),
            "entropy_top10": torch.Tensor(token_level_scores["entropy_top10"]).to(log_scores),
            "entropy_top15": torch.Tensor(token_level_scores["entropy_top15"]).to(log_scores),
            "weights": torch.Tensor(token_level_scores["weights"]).to(log_scores),
            "token_level_scores": token_level_scores
        })

    return output_dict
