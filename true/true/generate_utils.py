import logging
import os
from typing import Dict, Union, Tuple
 
import numpy as np
import torch
from scipy.stats import entropy

from torch.nn.functional import log_softmax
from true.default_values import DEFAULT_TOKEN_LEVEL_MEASURES, TOP_K

log = logging.getLogger()

def get_collect_fn(model_output):
    if type(model_output).__name__ == 'SampleEncoderDecoderOutput':
        return collect_sample_token_level_uncertainties
    return collect_token_level_uncertainties

def collect_sample_token_level_uncertainties(
        model_output,
        batch_size,
        num_return_sequences,
        vocab_size,
        pad_token_id,
        length_penalty=1.0,
        ensemble_uncertainties={}
    ):
    base_shape = [batch_size, num_return_sequences]
    seq_length = model_output['sequences'].shape[-1]

    seq_shape = base_shape + [seq_length]
    sequences = model_output['sequences'].reshape(seq_shape)[:, :, 1:]
    # 0 - iters
    # 1 - num_obs * num_ret_seq
    # 2 - vocab_size
    scores = torch.stack(model_output['scores']).permute(1, 0, 2)
    scores_shape = base_shape + [seq_length - 1, vocab_size]
    scores = scores.reshape(scores_shape)
    device = scores.device

    token_scores = torch.zeros(base_shape + [seq_length - 1]).to(device)

    token_measures = list(ensemble_uncertainties.keys()) + \
                     [f"entropy_top{k}" for k in TOP_K] + \
                     ['entropy']

    token_level_uncertainties = {
        key: torch.zeros(base_shape + [seq_length - 1]) for key in token_measures
    }

    output_uncertainties_reshaped = {
        key: torch.stack(ensemble_uncertainties[key], dim=-1).reshape(unc_shape) \
            for key in ensemble_uncertainties.keys()
    }
    
    seq_lengths = (model_output['sequences'] != pad_token_id).sum(dim=-1)
    seq_lengths = seq_lengths.reshape(base_shape).to(device)

    seq_penalty = (seq_lengths ** length_penalty)
    seq_penalty_unb = ((seq_lengths - 1) ** length_penalty)

    for obs_id in range(batch_size):
        for _iter in reversed(range(sequences.shape[-1])):
            for seq_i in range(num_return_sequences):
                index = (obs_id, seq_i, _iter)
                token = sequences[index]
                if token == pad_token_id:
                    continue
                else:
                    posterior_logs = log_softmax(scores[index], dim=-1)
                    token_scores[index] = posterior_logs[token]
                    posterior = posterior_logs.exp().detach().cpu()

                    entropies = {}
                    entropies['entropy'] = entropy(posterior)
                    entropies['entropy_top5'] = entropy(posterior.topk(5, dim=-1).values)
                    entropies['entropy_top10'] = entropy(posterior.topk(10, dim=-1).values)
                    entropies['entropy_top15'] = entropy(posterior.topk(15, dim=-1).values)
                    for key in token_measures:
                        if key in ['entropy', 'entropy_top5', 'entropy_top10', 'entropy_top15']:
                            ue = entropies[key]
                        else:
                            ue = output_uncertainties_reshaped[key][index]
                        token_level_uncertainties[key][index] = torch.tensor(ue)
    
    sequences_scores = token_scores.sum(dim=-1) / seq_penalty
    entropy_s = entropy(sequences_scores.exp().cpu().detach().numpy(), axis=-1)    

    for key in token_measures:
        token_level_uncertainties[key] = \
            token_level_uncertainties[key].sum(dim=-1).to(device)
        token_level_uncertainties[key] = \
            token_level_uncertainties[key] / seq_penalty_unb

    beam_weights = sequences_scores.exp() / sequences_scores.exp().sum(dim=-1, keepdim=True)
    token_level_uncertainties['beam_weights'] = beam_weights
    
    beam_scores_unb = sequences_scores * seq_penalty / seq_penalty_unb
    entropy_s_u = entropy(sequences_scores.exp().cpu().detach().numpy(), axis=-1)     

    token_level_uncertainties['scores_unbiased'] = beam_scores_unb
    beam_weights_unb = beam_scores_unb.exp() / beam_scores_unb.exp().sum(dim=-1, keepdim=True)
    token_level_uncertainties['weights'] = beam_weights_unb

    for key in token_level_uncertainties.keys():
        token_level_uncertainties[key] = \
            token_level_uncertainties[key].cpu().numpy()

    token_level_uncertainties['sequences_scores'] = sequences_scores.cpu().reshape(batch_size * num_return_sequences)
    token_level_uncertainties['entropy_s'] = entropy_s
    token_level_uncertainties['entropy_s_u'] = entropy_s_u

    return token_level_uncertainties


def collect_token_level_uncertainties(
        model_output,
        batch_size,
        beam_size,
        vocab_size,
        pad_token_id,
        length_penalty=1.0,
        ensemble_uncertainties={}
    ):
    beam_ids = model_output['beam_indices']
    seq_len = beam_ids.shape[-1]
    shape = (batch_size, beam_size, seq_len)
    beam_ids = beam_ids.reshape(shape)
    beam_ids = beam_ids[:, :, 1:]
    beam_ids_finished_mask = beam_ids == -1
    beam_ids = beam_ids % beam_size
    beam_ids[beam_ids_finished_mask] = -1
    
    token_measures = list(ensemble_uncertainties.keys()) + \
                     [f"entropy_top{k}" for k in TOP_K] + \
                     ['entropy']

    token_level_uncertainties = {
        key: torch.zeros(shape) for key in token_measures
    }
    
    aggregate_models = ('models_scores' in model_output and len(model_output['models_scores']) > 0)

    if aggregate_models:
        num_models = len(model_output['models_scores'][0])
        models_sequence_scores = torch.zeros(batch_size, num_models, beam_size, seq_len)

    if 'sequences_scores' not in model_output:
        sequence_scores = torch.zeros(batch_size, beam_size, seq_len)

    # For some reason, beam search can truncate generation iterations, so
    # seq len from beam_ids can be less than iterations steps number
    unc_length = len(model_output['scores'])
    unc_shape = (batch_size, beam_size, unc_length)
    output_uncertainties_reshaped = {
        key: torch.stack(ensemble_uncertainties[key], dim=-1).reshape(unc_shape) \
            for key in ensemble_uncertainties.keys()
    }
    
    device = beam_ids.device
    seq_lengths = (model_output['sequences'] != pad_token_id).sum(dim=-1)
    seq_lengths = seq_lengths.reshape(batch_size, beam_size).to(device)

    seq_penalty = (seq_lengths ** length_penalty)
    seq_penalty_unb = ((seq_lengths - 1) ** length_penalty)

    sequences = model_output['sequences'].reshape(shape)[:, :, 1:]
        
    for obs_id in range(batch_size):
        for _iter in reversed(range(beam_ids.shape[-1])):
            iter_beam_ids = beam_ids[obs_id, :, _iter]
            for seq_i, beam_id in enumerate(iter_beam_ids):
                if beam_id == -1:
                    continue
                else:
                    if aggregate_models:
                        token = sequences[obs_id, seq_i, _iter]
                        for i, model_logits in enumerate(model_output['models_scores'][_iter]):
                            model_logits = model_logits.reshape(batch_size, beam_size, vocab_size)
                            models_sequence_scores[obs_id, i, seq_i, _iter] = \
                               model_logits[obs_id, beam_id, token] 
                            
                    posterior = model_output['scores'][_iter].reshape(batch_size,
                                                                      beam_size,
                                                                      vocab_size)[obs_id, beam_id].detach().cpu().exp()
                    entropies = {}
                    entropies['entropy'] = entropy(posterior)
                    entropies['entropy_top5'] = entropy(posterior.topk(5, dim=-1).values)
                    entropies['entropy_top10'] = entropy(posterior.topk(10, dim=-1).values)
                    entropies['entropy_top15'] = entropy(posterior.topk(15, dim=-1).values)
                    for key in token_measures:
                        if key in ['entropy', 'entropy_top5', 'entropy_top10', 'entropy_top15']:
                            ue = entropies[key]
                        else:
                            ue = output_uncertainties_reshaped[key][obs_id, beam_id, _iter]
                        token_level_uncertainties[key][obs_id, seq_i, _iter] = torch.tensor(ue)

    for key in token_measures:
        token_level_uncertainties[key] = \
            token_level_uncertainties[key].sum(dim=-1).to(device)
        token_level_uncertainties[key] = \
            token_level_uncertainties[key] / seq_penalty_unb
    
    if aggregate_models:
        models_sequence_scores = models_sequence_scores.sum(dim=-1).to(device) / seq_penalty_unb
        token_level_uncertainties['log_probas'] = models_sequence_scores
        token_level_uncertainties['probas'] = models_sequence_scores.exp()

    beam_scores = model_output['sequences_scores'].reshape(batch_size, beam_size)
    entropy_s = entropy(beam_scores.exp().cpu().detach().numpy(), axis=-1)    
    beam_weights = beam_scores.exp() / beam_scores.exp().sum(dim=-1, keepdim=True)
    token_level_uncertainties['beam_weights'] = beam_weights
    
    beam_scores_unb = beam_scores * (seq_penalty / seq_penalty_unb)
    entropy_s_u = entropy(beam_scores_unb.exp().cpu().detach().numpy(), axis=-1)    
    token_level_uncertainties['scores_unbiased'] = beam_scores_unb
    beam_weights_unb = beam_scores_unb.exp() / beam_scores_unb.exp().sum(dim=-1, keepdim=True)
    token_level_uncertainties['weights'] = beam_weights_unb

    for key in token_level_uncertainties.keys():
        token_level_uncertainties[key] = \
            token_level_uncertainties[key].cpu().numpy()

    token_level_uncertainties['entropy_s'] = entropy_s
    token_level_uncertainties['entropy_s_u'] = entropy_s_u

    return token_level_uncertainties

def update_token_level_scores(scores, batch_scores):
    for key in scores:
        if scores[key] is None:
            scores[key] = batch_scores[key]
        else:
            scores[key] = np.r_[scores[key], batch_scores[key]]
    return scores


def update_iter_scores(scores, cur_iter_scores, max_gen_len=None):
    if scores is None:
        scores = cur_iter_scores
    if (max_gen_len is not None) and (len(scores) < max_gen_len - 1):
        scores = pad_scores_tensor(scores, max_gen_len - 1 - len(scores))
    else:
        if len(cur_iter_scores) < len(scores):
            cur_iter_scores = pad_scores_tensor(
                cur_iter_scores, len(scores) - len(cur_iter_scores)
            )
        for i, scores_matrix in enumerate(scores):
            scores[i] = torch.cat([scores[i], cur_iter_scores[i]], dim=0)
    return scores


def pad_scores_tensor(scores_tensor, pad_value):
    tensor_to_add = torch.zeros(*scores_tensor[0].shape) - torch.inf
    if len(tensor_to_add.shape) >= 2:
        tensor_to_add[:, 2] = 0
    tensor_to_add = tensor_to_add.to(scores_tensor[0])
    for i in range(pad_value):
        scores_tensor += (tensor_to_add,)
    return scores_tensor


def restore_token_level_data(
    output: Dict,  # dict with `sequences`, `scores`, `model_config` and token-level measures keys
    num_obs: int = None,
    num_beams: int = None,
    max_gen_len=None,
    softmax_t: Union[int, float] = 1,
    topk: Tuple[int] = (5, 10, 15),
    calculate_token_level_scores: bool = True,
    model_config=None,
    length_penalty: Union[float, int] = 1.0,
    **kwargs,
):
    # TODO: add a parameter to config
    device = os.environ.get("DEVICE_FOR_DATA_RESTORING", "cuda")
    scores = [x.to(device) for x in output["scores"]]
    sequences = output["sequences"].to(device)
    model_config = model_config if model_config is not None else output["model_config"]
    #$token_level_keys = DEFAULT_TOKEN_LEVEL_MEASURES
    token_level_keys = [f"pe_{measure}" for measure in DEFAULT_TOKEN_LEVEL_MEASURES]

    if calculate_token_level_scores:
        token_level_data = {}
        for key in token_level_keys:
            token_level_data[key] = [x.to(device) for x in output[key]]

    if num_obs is None:
        num_obs = sequences.shape[0]
    if num_beams is None:
        num_beams = sequences.shape[0] // num_beams
    if max_gen_len is None:
        # fix for MBART generation; sometimes something strange happens with the length of scores from MBART
        len_difference = (
            len(output["scores"]) - sequences.reshape(num_obs, num_beams, -1).shape[-1]
        )
        if len_difference >= 0:
            output["scores"] = output["scores"][: -1 - len_difference]
            scores = scores[: -1 - len_difference]
        max_gen_len = len(output["scores"]) + 1

    beam_scores = None
    num_beams_plus_1 = num_beams + 1
    eos_id = model_config.eos_token_id

    beams = (
        torch.LongTensor([model_config.decoder_start_token_id])
        .repeat(num_obs, num_beams, 1)
        .to(scores[0].device)
    )
    finished_beams = [[] for _ in range(num_obs)]
    finished_beam_scores = [[] for _ in range(num_obs)]
    finished_unbiased_beam_scores = [[] for _ in range(num_obs)]
    finished_entropies = [[] for _ in range(num_obs)]
    finished_entropies_topk = {k: [[] for _ in range(num_obs)] for k in topk}
    if calculate_token_level_scores:
        for key in token_level_keys:
            token_level_data[f"finished_{key}"] = [[] for _ in range(num_obs)]

    eos_token_id = model_config.eos_token_id
    eos_tensor = torch.LongTensor([eos_token_id]).to(beams)
    pad = torch.LongTensor([model_config.pad_token_id]).to(beams)
    sequences_reshaped = sequences.reshape(num_obs, num_beams, -1)
    sequences_scores = (
        output["sequences_scores"].cpu().numpy().reshape(num_obs, num_beams)
    )

    for i, iter_scores in enumerate(scores):
        iter_scores = iter_scores.reshape(num_obs, num_beams, -1)
        # num_beams + 1 since we can have EOS and num_beams continuations
        vals, idx = iter_scores.topk(num_beams_plus_1, dim=-1)
        if beam_scores is None:
            # In this case all the beams for one instance coincide, hence can take the zero-th beams
            beams = torch.cat([beams, idx[:, 0, :-1, None]], dim=-1)
            beam_scores = vals[:, 0, :-1][:, :, None]
            entropies = vals[:, 0, :-1]
            entropies_topk = {k: vals[:, 0, :-1] for k in topk}
            if calculate_token_level_scores:
                for key in token_level_keys:
                    token_level_data[f"processing_{key}"] = token_level_data[key][
                        i
                    ].reshape(num_obs, num_beams)[:, 0, None].repeat(1, num_beams)

        else:
            if calculate_token_level_scores:
                for key in token_level_keys:
                    token_level_data[f"iter_{key}"] = token_level_data[key][i].reshape(
                        num_obs, num_beams
                    )
            scores_with_beam_scores = (beam_scores + vals).reshape(num_obs, -1)
            # Entropies part start
            iter_scores_clipped = iter_scores.clip(-1e7, -1e-7)
            cur_beam_entropies = -(iter_scores_clipped * iter_scores_clipped.exp()).sum(
                -1
            )
            cur_beam_entropies_topk = {}
            for k in topk:
                iter_scores_topk = iter_scores_clipped.topk(k, dim=-1).values
                cur_beam_entropies_topk[k] = -(
                    iter_scores_topk * iter_scores_topk.exp()
                ).sum(-1)

            # Entropies part end

            for i_batch, batch_idx in enumerate(idx):
                for i_beam, beam_idx in enumerate(batch_idx):
                    is_over = eos_id in beam_idx
                    if is_over:
                        idx_is_over_inside_beam = (beam_idx == eos_id).int().argmax()
                        finished_beams[i_batch].append(
                            torch.cat([beams[i_batch, i_beam], eos_tensor])
                        )
                        idx_id_over = (
                            i_beam * num_beams_plus_1 + idx_is_over_inside_beam
                        )
                        seq_len_penalty = (
                            len(finished_beams[i_batch][-1]) - 1
                        ) ** length_penalty
                        seq_len_penalty_unb = (
                            len(finished_beams[i_batch][-1]) - 2
                        ) ** length_penalty
                        finished_beam_scores[i_batch].append(
                            scores_with_beam_scores[i_batch, idx_id_over].item()
                            / (seq_len_penalty)
                        )
                        finished_unbiased_beam_scores[i_batch].append(
                            scores_with_beam_scores[i_batch, idx_id_over].item()
                            / (seq_len_penalty_unb)
                        )
                        finished_entropies[i_batch].append(
                            (
                                entropies[i_batch, i_beam]
                                + cur_beam_entropies[i_batch, i_beam]
                            )
                            / seq_len_penalty_unb
                        )
                        for k in topk:
                            finished_entropies_topk[k][i_batch].append(
                                (
                                    entropies_topk[k][i_batch, i_beam]
                                    + cur_beam_entropies_topk[k][i_batch, i_beam]
                                )
                                / seq_len_penalty_unb
                            )
                        if calculate_token_level_scores:
                            for key in token_level_keys:
                                token_level_data[f"finished_{key}"][i_batch].append(
                                    (
                                        token_level_data[f"processing_{key}"][
                                            i_batch, i_beam
                                        ]
                                        + token_level_data[f"iter_{key}"][
                                            i_batch, i_beam
                                        ]
                                    )
                                    / seq_len_penalty_unb
                                )
                        scores_with_beam_scores[i_batch, idx_id_over] = -torch.inf
            # Need to avoid `torch.topk` since topk works incorrect for equal values:
            # https://github.com/pytorch/pytorch/issues/27542
            top_nbeams_idx = (
                scores_with_beam_scores.cpu()
                .argsort(dim=-1, descending=True)[:, :num_beams]
                .to(device)
            )
            top_nbeams_vals = scores_with_beam_scores.gather(-1, top_nbeams_idx)
            # top_nbeams = scores_with_beam_scores.topk(num_beams, dim=-1)
            # top_nbeams_vals, top_nbeams_idx = top_nbeams
            top_nbeams_idx_beam = top_nbeams_idx // num_beams_plus_1
            next_word_idx = (
                idx.reshape(num_obs, -1)
                .gather(index=top_nbeams_idx, dim=-1)
                .unsqueeze(-1)
            )
            # Using dim== -2 because dim == -1 is empty (for scores / over flag) or contains sequence (for beams)
            beams = beams.gather(
                index=top_nbeams_idx_beam.unsqueeze(-1).repeat(1, 1, beams.shape[-1]),
                dim=-2,
            )
            beams = torch.cat([beams, next_word_idx], dim=-1)

            beam_scores = top_nbeams_vals.unsqueeze(-1)

            entropies = (entropies + cur_beam_entropies).gather(
                index=top_nbeams_idx_beam, dim=-1
            )
            for k in entropies_topk:
                entropies_topk[k] = (
                    entropies_topk[k] + cur_beam_entropies_topk[k]
                ).gather(index=top_nbeams_idx_beam, dim=-1)
            if calculate_token_level_scores:
                for key in token_level_keys:
                    token_level_data[f"processing_{key}"] = (
                        token_level_data[f"processing_{key}"]
                        + token_level_data[f"iter_{key}"]
                    ).gather(index=top_nbeams_idx_beam, dim=-1)
    # Add beams that were still "active" on the last generation iteration
    for i_batch in range(num_obs):
        for i_beam in range(num_beams):
            # Only add if the score != -inf
            if beam_scores[i_batch, i_beam] == -torch.inf:
                continue
            finished_beams[i_batch].append(beams[i_batch, i_beam])
            seq_len_penalty = len(scores) ** length_penalty
            seq_len_penalty_unb = (len(scores) - 1) ** length_penalty
            finished_beam_scores[i_batch].append(
                beam_scores[i_batch, i_beam].item() / seq_len_penalty
            )
            finished_unbiased_beam_scores[i_batch].append(
                beam_scores[i_batch, i_beam].item() / seq_len_penalty_unb
            )
            finished_entropies[i_batch].append(
                entropies[i_batch, i_beam] / seq_len_penalty_unb
            )
            for k in topk:
                finished_entropies_topk[k][i_batch].append(
                    entropies_topk[k][i_batch, i_beam] / seq_len_penalty_unb
                )
            if calculate_token_level_scores:
                for key in token_level_keys:
                    token_level_data[f"finished_{key}"][i_batch].append(
                        token_level_data[f"processing_{key}"][i_batch, i_beam]
                        / seq_len_penalty_unb
                    )

        # Todo: move to the line before the last when fixing the argsort (what?)

        for i, beam in enumerate(finished_beams[i_batch]):
            len_beam = len(finished_beams[i_batch][i])
            finished_beams[i_batch][i] = torch.cat(
                [beam, pad.repeat(max_gen_len - len_beam)]
            )

        # Get idx of best beams based on the scores
        scores_obtained = np.tile(
            finished_beam_scores[i_batch], (num_beams, 1)
        ).T.astype(np.float32)
        scores_true = np.tile(sequences_scores[i_batch], (len(scores_obtained), 1))
        idx_best = np.isclose(scores_obtained, scores_true).argmax(0)

        finished_beam_scores[i_batch] = torch.Tensor(
            [finished_beam_scores[i_batch][i] for i in idx_best]
        ).to(beam_scores)
        finished_unbiased_beam_scores[i_batch] = torch.Tensor(
            [finished_unbiased_beam_scores[i_batch][i] for i in idx_best]
        ).to(beam_scores)
        finished_entropies[i_batch] = torch.Tensor(
            [finished_entropies[i_batch][i] for i in idx_best]
        ).to(beam_scores)
        for k in topk:
            finished_entropies_topk[k][i_batch] = torch.Tensor(
                [finished_entropies_topk[k][i_batch][i] for i in idx_best]
            ).to(beam_scores)
        if calculate_token_level_scores:
            for key in token_level_keys:
                token_level_data[f"finished_{key}"][i_batch] = torch.Tensor(
                    [token_level_data[f"finished_{key}"][i_batch][i] for i in idx_best]
                ).to(beam_scores)

    beam_scores = torch.stack(finished_beam_scores)
    beam_scores_unbiased = torch.stack(finished_unbiased_beam_scores)
    entropies = torch.stack(finished_entropies)

    entropies_topk_out = {}
    for k in topk:
        entropies_topk_out[f"entropy_top{k}"] = (
            torch.stack(finished_entropies_topk[k]).cpu().numpy()
        )

    beam_scores_exp = beam_scores.exp() ** softmax_t
    beam_scores_unb_exp = beam_scores_unbiased.exp() ** softmax_t
    beam_weights = beam_scores_exp / beam_scores_exp.sum(-1, keepdims=True)
    beam_weights_unbiased = beam_scores_unb_exp / beam_scores_unb_exp.sum(
        -1, keepdims=True
    )

    out_dict = dict(
        beams=sequences_reshaped.cpu().numpy(),
        beam_scores=beam_scores.cpu().numpy(),
        scores_unbiased=beam_scores_unbiased.cpu().numpy(),
        beam_weights=beam_weights.cpu().numpy(),
        weights=beam_weights_unbiased.cpu().numpy(),
        entropy=entropies.cpu().numpy(),
    )
    out_dict.update(entropies_topk_out)

    if calculate_token_level_scores:
        token_level_out_dict = {}
        for key in token_level_keys:
            token_level_out_dict[key] = (
                torch.stack(token_level_data[f"finished_{key}"]).cpu().numpy()
            )
        out_dict.update(token_level_out_dict)

    # Assert no errors have been made (e.g. when two beams had the same probability when casted to float32)
    # true_sequences = sequences_reshaped.cpu().numpy()
    # out_dict = fix_broken_beams(
    #     true_sequences, out_dict, topk, calculate_token_level_scores
    # )
    return out_dict


def fix_broken_beams(
    true_sequences,
    out_dict,
    topk: Tuple[int] = (5, 10, 15),
    update_token_level_scores: bool = True,
):
    keys_to_update = ["beams"] + [f"entropy_top{k}" for k in topk]
    if update_token_level_scores:
        keys_to_update += DEFAULT_TOKEN_LEVEL_MEASURES
    # Find broken beams
    all_beams_coincide = (true_sequences == out_dict["beams"]).all(1).all(1)
    broken_idx = np.argwhere(~all_beams_coincide).ravel()
    # Fix
    for idx in broken_idx:
        true_order = []
        for i, idx_true_seq in enumerate(true_sequences[idx]):
            added = False
            for j, idx_gen_seq in enumerate(out_dict["beams"][idx]):
                if (idx_true_seq == idx_gen_seq).all():
                    true_order.append(j)
                    added = True
                    break
            if not added:
                true_order.append(-1)
        num_not_added = true_order.count(-1)
        if num_not_added == 0:
            not_added_idx = iter(np.setdiff1d(range(len(true_order)), true_order))
            for i, idx_true_seq in enumerate(true_order):
                if idx_true_seq == -1:
                    true_order[i] = next(not_added_idx)

        for key in keys_to_update:
            out_dict[key][idx] = out_dict[key][idx][true_order]
    return out_dict


def from_text_to_id(mapping, x):
    try:
        return list(mapping[("en", x)])[0]
    except:
        return "None"
