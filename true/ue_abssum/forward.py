import gc
from typing import Union, List

import torch
from datasets import Dataset
from torch.nn import NLLLoss
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
)

from ue_abssum.default_values import DEFAULT_TOKEN_LEVEL_MEASURES
from ue_abssum.forward_utils import calculate_token_level_measures
from ue_abssum.tokenize_data import tokenize_data


def forward(
    model: Union[AutoModelForSeq2SeqLM, List[AutoModelForSeq2SeqLM]],
    data: Dataset,
    tokenizer: AutoTokenizer,
    seed,
    is_tokenized=False,
    data_config=None,
    to_numpy: bool = False,
    to_eval_mode: bool = True,
    batch_size: int = None,
    max_seq_len: int = None,
    mc: bool = False,
    mc_iterations: int = 1,
    **kwargs,
):
    """

    :param model:
    :param data:
    :param tokenizer:
    :param is_tokenized:
    :param data_config:
    :param to_numpy:
    :param to_eval_mode:
    :param batch_size:
    :return:
    """
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
    assert "labels" in data.features, "Labels must be provided when using forward!"

    # If dealing with a single model, make simple forward pass through it
    if not hasattr(model, "__iter__"):
        output = _model_forward_loop(
            model=model,
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            to_eval_mode=to_eval_mode,
        )
        max_num_labels = output["num_labels"].max()
        output["probas"] = output["probas"][:, :max_num_labels]
        output["log_probas"] = output["log_probas"][:, :max_num_labels]
        output["labels"] = output["labels"][:, :max_num_labels]
    else:
        output = _ensemble_forward_loop(
            models=model,
            mc = mc,
            mc_iterations = mc_iterations,
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            to_eval_mode=to_eval_mode,
            max_seq_len=max_seq_len,
            seed=seed
        )
        output["log_probas_iter"] = output["log_probas_iter"][:, :, :max_seq_len]
        output["labels"] = output["labels"][:, :max_seq_len]
        output["mask"] = output["mask"][:, :max_seq_len]

    if to_numpy:
        output = {
            key: (
                val.cpu().detach().numpy()
                if not isinstance(val, dict)  # if key == 'token_level_measures'
                else {
                    key_inner: val_inner.cpu().detach().numpy()
                    for key_inner, val_inner in val.items()
                }
            )
            for key, val in output.items()
        }
    torch.cuda.empty_cache()
    gc.collect()
    return output


def _model_forward_loop(
    model: Union[
        AutoModelForSeq2SeqLM,
        BartForConditionalGeneration,
        PegasusForConditionalGeneration,
    ],
    data: Dataset,
    tokenizer: AutoTokenizer,
    batch_size=None,
    to_eval_mode=True,
):
    """
    Implemented for seq2seq tasks
    Returns: dict with sequences of ids and their scores

    """
    torch.cuda.empty_cache()
    if batch_size is None:
        batch_size = 50
    assert "labels" in data.features, "Labels must be provided when using forward!"

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest"),
    )

    if to_eval_mode:
        model.eval()
    device = model.device
    num_obs = len(data)
    max_seq_len = tokenizer.model_max_length
    pad_idx = dataloader.collate_fn.label_pad_token_id
    log_probas = torch.zeros((num_obs, max_seq_len), dtype=torch.float32, device=device)
    losses = torch.empty(num_obs, dtype=torch.float32, device=device)
    labels = (
        torch.zeros((num_obs, max_seq_len), dtype=torch.int32, device=device) + pad_idx
    )
    num_labels = torch.empty(num_obs, dtype=torch.int16, device=device)

    loss_fn = NLLLoss(reduction="none", ignore_index=pad_idx)

    with torch.no_grad():
        start = 0
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            end = start + len(batch["input_ids"])
            output = model(**batch)

            # Calculate scores without the first token since it is always <CLS>
            batch_labels = batch["labels"]  # batch_size x seq_len
            labels_len = (batch_labels != pad_idx).sum(-1)
            logits = output.logits.view(
                -1, model.config.vocab_size
            )  # batch_size * seq_len x vocab_size
            log_probas_batch = log_softmax(logits, dim=-1)
            loss_tokens = loss_fn(log_probas_batch, batch_labels.view(-1))
            loss_tokens = loss_tokens.view(batch_labels.shape[0], -1)
            loss_texts = loss_tokens.sum(dim=1) / labels_len  # batch_size

            log_probas[start:end, : batch_labels.shape[1]].copy_(
                -loss_tokens, non_blocking=True
            )
            losses[start:end].copy_(loss_texts, non_blocking=True)
            num_labels[start:end].copy_(labels_len, non_blocking=True)
            labels[start:end, : batch_labels.shape[1]].copy_(
                batch_labels, non_blocking=True
            )
            start = end
    # num_obs x model_max_length
    probas = log_probas.exp()
    scores = (-losses).exp()
    mask = torch.Tensor(data["labels"]) != -100

    return {
        "scores": scores,
        "losses": losses,
        "probas": probas,
        "mask": mask,
        "log_probas": log_probas,
        "num_labels": num_labels,
        "labels": labels,
    }


def _ensemble_forward_loop(
    models: List[
        Union[
            AutoModelForSeq2SeqLM,
            BartForConditionalGeneration,
            PegasusForConditionalGeneration,
        ]
    ],
    data: Dataset,
    tokenizer: AutoTokenizer,
    seed,
    mc=False,
    mc_iterations=1,
    batch_size=None,
    to_eval_mode=True,
    topk_for_entropy: List[int] = (5, 10, 15),
    max_seq_len: int = None,
    to_float64: bool = True,
):
    """
    Implemented for seq2seq tasks
    Returns: dict with sequences of ids and their scores

    """
    torch.cuda.empty_cache()
    if batch_size is None:
        batch_size = 50
    assert "labels" in data.features, "Labels must be provided when using forward!"

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest"),
    )

    if to_eval_mode:
        for model in models:
            model.eval()
    device = models[0].device
    if mc:
        ens_size = mc_iterations
    else:
        ens_size = len(models)
    num_obs = len(data)
    max_seq_len = max_seq_len or tokenizer.model_max_length
    # Get vocabulaty size
    for param in models[0].parameters():
        break
    vocab_size = param.shape[0]
    pad_idx = tokenizer.pad_token_id  # dataloader.collate_fn.label_pad_token_id
    log_probas_iter = torch.zeros(
        (ens_size, num_obs, max_seq_len), dtype=torch.float32, device=device
    )
    # Token-level uncertainty
    token_level_measures = {}
    for key in DEFAULT_TOKEN_LEVEL_MEASURES:
        token_level_measures[key] = torch.zeros(
            num_obs, dtype=torch.float32, device=device
        )
    for topk in topk_for_entropy:
        token_level_measures[f"entropy_topk{topk}"] = torch.zeros(
            num_obs, dtype=torch.float32, device=device
        )
    labels = (
        torch.zeros((num_obs, max_seq_len), dtype=torch.int32, device=device) + pad_idx
    )

    loss_fn = NLLLoss(reduction="none", ignore_index=pad_idx)
    models_scores = []
    with torch.no_grad():
        start = 0
        for batch in tqdm(dataloader):
            # Placeholders for token-level stats
            batch_size, seq_len = batch["labels"].shape
            crude_logits = torch.zeros(
                (ens_size, batch_size, seq_len, vocab_size),
                dtype=torch.float32,
                device=device,
            )
            log_probas = torch.zeros(
                (ens_size, batch_size, seq_len, vocab_size),
                dtype=torch.float32,
                device=device,
            )
            label_log_probas = torch.zeros(
                (ens_size, batch_size, seq_len), dtype=torch.float32, device=device
            )
            # Calculate batch output
            batch = {k: v.to(device) for k, v in batch.items()}
            end = start + len(batch["input_ids"])
            outputs = []
            if mc:
                global_seed = seed
                for i in range(mc_iterations):
                    torch.manual_seed(global_seed + i)
                    outputs.append(models[0](**batch))
                torch.manual_seed(global_seed)
            else:
                for model in models:
                    outputs.append(model(**batch))
            # Calculate scores without the first token since it is always <CLS>
            batch_labels = batch["labels"]  # batch_size x seq_len
            # To avoid selecting -100th element in the future
            batch_labels[batch_labels == -100] = pad_idx  # batch_size x seq_len
            labels_mask = (batch_labels != pad_idx)[:, 1:]
            batch_labels_raveled = batch_labels.reshape(-1).unsqueeze(
                1
            )  # batch_size * seq_len
            for i in range(ens_size):
                if mc:
                    model = models[0]
                else:
                    model = models[i]

                logits = outputs[i].logits.view(
                    -1, model.config.vocab_size
                )  # batch_size * seq_len x vocab_size
                log_probas_batch = log_softmax(
                    logits, dim=-1
                )  # batch_size * seq_len x vocab_size
                loss_tokens = loss_fn(log_probas_batch, batch_labels.view(-1)).view(
                    batch_size, -1
                )  # batch_size x seq_len
                loss_tokens[:, 1:][~labels_mask] = 0
                log_probas_iter[i, start:end, :seq_len].copy_(
                    -loss_tokens, non_blocking=True
                )  # batch_size x seq_len

                label_log_probas_batch = log_probas_batch.gather(
                    dim=1, index=batch_labels_raveled
                )
                label_log_probas_batch[batch_labels_raveled == pad_idx] = 0
                label_log_probas_batch = label_log_probas_batch.reshape(
                    len(outputs[i].logits), -1
                )

                # Reshape back to original shape
                log_probas_batch = log_probas_batch.reshape(
                    *batch["labels"].shape, -1
                )  # batch_size x seq_len x vocab_size
                # Save model results
                log_probas[i].copy_(log_probas_batch)
                label_log_probas[i].copy_(label_log_probas_batch)

            token_level_scores = calculate_token_level_measures(
                log_probas, label_log_probas, labels_mask, to_float64
            )
            for key in token_level_scores:
                token_level_measures[key][start:end].copy_(token_level_scores[key])
            labels[start:end, : batch_labels.shape[1]].copy_(
                batch_labels, non_blocking=True
            )
            start = end
            models_scores.append(crude_logits.cpu())
            del log_probas, label_log_probas, labels_mask, logits, loss_tokens
            torch.cuda.empty_cache()

    mask = torch.Tensor(data["labels"]).to(device) != -100  # num_obs x seq_len
    num_labels = mask.sum(-1)  # num_obs
    log_probas = (
        log_probas_iter.sum(-1) / num_labels
    )  # num_models x num_obs x num_beams
    probas = log_probas.exp()  # num_models x num_obs x num_beams
    
    return {
        "probas": probas,
        "log_probas": log_probas,
        "log_probas_iter": log_probas_iter,
        "labels": labels,
        "mask": mask,
        "num_labels": num_labels,
        "token_level_measures": token_level_measures,
        "outputs": outputs
    }
