from typing import Union

import torch
from datasets import Dataset as ArrowDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ue_abssum.model_modifications import (
    get_model_without_cls_layer,
    _get_dim,
    ModelForFeaturesExtraction,
)
from ue_abssum.tokenize_data import tokenize_data


def get_embeddings(
    model,
    dataloader_or_data: Union[DataLoader, ArrowDataset],
    prepare_model: bool = True,
    use_activation: bool = False,
    use_spectralnorm: bool = False,
    to_eval_mode: bool = True,
    to_numpy: bool = False,
    data_is_tokenized: bool = False,
    batch_size: int = 100,
    use_automodel: bool = False,
    use_averaging: bool = False,
    hidden_state: str = "encoder",
    ignore_padding: bool = False,
    generation_max_length: int = 100,
    extraction_method: str = "forward",
    all_layers: bool = False,
    aggregation_method: str = "mean",
    **tokenization_kwargs,
):
    if prepare_model:
        model_without_cls_layer = get_model_without_cls_layer(
            model, use_activation, use_spectralnorm
        )
    else:
        model_without_cls_layer = model

    device = next(model.parameters()).device

    if not isinstance(dataloader_or_data, DataLoader):
        if not data_is_tokenized:
            dataloader_or_data = tokenize_data(
                data=dataloader_or_data, **tokenization_kwargs, padding=True
            )
        dataloader_or_data = DataLoader(
            dataloader_or_data,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenization_kwargs["tokenizer"]
            ),
            pin_memory=(str(device).startswith("cuda")),
        )

    num_obs = len(dataloader_or_data.dataset)
    dim = _get_dim(model_without_cls_layer)
    if to_eval_mode:
        model_without_cls_layer.eval()

    embeddings = torch.empty((num_obs, dim), dtype=torch.float, device=device)
    embeddings_decoder = torch.empty((num_obs, dim), dtype=torch.float, device=device)

    if isinstance(model_without_cls_layer, ModelForFeaturesExtraction):
        possible_input_keys = model_without_cls_layer[
            0
        ].model.forward.__code__.co_varnames
    else:
        possible_input_keys = model_without_cls_layer.forward.__code__.co_varnames
    possible_input_keys = list(possible_input_keys) + ["input_ids", "attention_mask"]

    generation_kwargs = {}
    if "mbart" in model.config._name_or_path:
        model.config.decoder_start_token_id = tokenization_kwargs[
            "tokenizer"
        ].lang_code_to_id[tokenization_kwargs["tokenizer"].tgt_lang]
        generation_kwargs[
            "decoder_start_token_id"
        ] = model.config.decoder_start_token_id
    with torch.no_grad():
        torch.cuda.empty_cache()
        start = 0
        for batch in tqdm(dataloader_or_data, desc="Embeddings created"):
            batch = {
                k: v.to(device) for k, v in batch.items() if k in possible_input_keys
            }
            if extraction_method == "forward":
                predictions = model_without_cls_layer(
                    **batch,
                    output_hidden_states=True,
                )
                batch_embeddings, _ = get_embeddings_from_output(
                    predictions,
                    model,
                    batch,
                    hidden_state,
                    ignore_padding,
                    extraction_method,
                    use_activation,
                    use_automodel,
                    use_averaging,
                    all_layers,
                    aggregation_method,
                )
            elif extraction_method == "generate":
                output = model.generate(
                    **batch,
                    output_scores=True,
                    max_length=generation_max_length,
                    min_length=3,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    num_beams=1,
                    **generation_kwargs,
                )
                batch_embeddings, batch_embeddings_decoder = get_embeddings_from_output(
                    output,
                    None,
                    batch,
                    hidden_state,
                    ignore_padding,
                    extraction_method,
                    use_activation,
                    use_automodel,
                    use_averaging,
                    all_layers,
                    aggregation_method,
                )
            else:
                raise NotImplementedError

            end = start + len(batch["input_ids"])  # len(batch[list(batch.keys())[0]])
            if extraction_method != "generate":
                embeddings[start:end].copy_(batch_embeddings, non_blocking=True)
            if "encoder" in hidden_state:
                embeddings[start:end].copy_(batch_embeddings, non_blocking=True)
            if "decoder" in hidden_state:
                embeddings_decoder[start:end].copy_(
                    batch_embeddings_decoder, non_blocking=True
                )

            start = end

    embeddings_dict = {}
    if extraction_method == "generate":
        if "encoder" in hidden_state:
            embeddings_dict["encoder"] = embeddings
        if "decoder" in hidden_state:
            embeddings_dict["decoder"] = embeddings_decoder
    elif extraction_method == "forward":
        embeddings_dict["encoder"] = embeddings

    if to_numpy:
        embeddings_dict = {
            k: embeddings_dict[k].cpu().detach().numpy() for k in embeddings_dict.keys()
        }

    return embeddings_dict

def aggregate(x, aggregation_method, axis):
    if aggregation_method == "max":
        return x.max(axis=axis).values
    elif aggregation_method == "mean":
        return x.mean(axis=axis)
    elif aggregation_method == "sum":
        return x.sum(axis=axis)

def get_embeddings_from_output(
    output,
    model,
    batch,
    hidden_state: str = "encoder",
    ignore_padding: bool = False,
    extraction_method: str = "forward",
    use_activation: bool = False,
    use_automodel: bool = True,
    use_averaging: bool = False,
    all_layers: bool = False,
    aggregation_method: str = "mean",
    **kwargs,
):
    # TODO: make smarter
    batch_embeddings = None
    batch_embeddings_decoder = None
    batch_size = len(batch["input_ids"])
    deepfool_scores = torch.zeros(batch_size)
    if isinstance(model, ModelForFeaturesExtraction):
        batch_embeddings = predictions
    elif (extraction_method == "forward") and (use_automodel and not use_activation):
        if use_averaging:
            batch_embeddings = predictions.last_hidden_state.mean(1)
        else:
            batch_embeddings = predictions.last_hidden_state[:, 0]

    elif (extraction_method == "generate") and (use_automodel and not use_activation):
        if use_averaging:
            if "decoder" in hidden_state:
                decoder_hidden_states = torch.stack(
                    [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                )
                if all_layers:
                    agg_decoder_hidden_states = decoder_hidden_states[:, :, :, 0].mean(axis=1)
                else:
                    agg_decoder_hidden_states = decoder_hidden_states[:, -1, :, 0]

                    batch_embeddings_decoder = aggregate(agg_decoder_hidden_states, aggregation_method, axis=0)
                    batch_embeddings_decoder = batch_embeddings_decoder.reshape(batch_size, -1, agg_decoder_hidden_states.shape[-1])[:, 0]
            if "encoder" in hidden_state:
                mask = batch["attention_mask"][:, :, None]
                seq_lens = batch["attention_mask"].sum(-1)[:, None]
                if all_layers:
                    encoder_embeddings = aggregate(torch.stack(output.encoder_hidden_states), "mean", axis=0) * mask
                else:
                    encoder_embeddings = output.encoder_hidden_states[-1] * mask
                if ignore_padding:
                    if aggregation_method == "mean":
                        batch_embeddings = (
                            encoder_embeddings
                        ).sum(1) / seq_lens
                    else:
                        batch_embeddings = aggregate(encoder_embeddings, aggregation_method, axis=1)
                else:
                    batch_embeddings = aggregate(encoder_embeddings, aggregation_method, axis=1)
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
        else:
            if "decoder" in hidden_state:
                decoder_hidden_states = torch.stack(
                    [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                )
                last_decoder_hidden_states = decoder_hidden_states[-1, -1, :, 0]
                batch_embeddings_decoder = last_decoder_hidden_states.reshape(
                    batch_size, -1, last_decoder_hidden_states.shape[-1]
                )[:, 0]
            if "encoder" in hidden_state:
                batch_embeddings = predictions.encoder_hidden_states[-1][:, 0]
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
    elif "pooler_output" in predictions.keys():
        batch_embeddings = predictions.pooler_output
    elif "last_hidden_state" in predictions.keys():
        batch_embeddings = predictions.last_hidden_state
    else:
        raise NotImplementedError

    return batch_embeddings, batch_embeddings_decoder
