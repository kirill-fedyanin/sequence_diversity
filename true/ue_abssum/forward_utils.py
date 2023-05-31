from math import log

import torch
from scipy.stats import entropy


def calculate_token_level_measures(
    log_probas: torch.Tensor,  # ens_size x batch_size x seq_len x vocab_size
    label_log_probas: torch.Tensor,  # ens_size x batch_size x seq_len
    labels_mask: torch.BoolTensor,  # batch_size x seq_len
    to_float64: bool = True,
    entropy_topk: tuple = (5, 10, 15),
):
    output = {}
    reference_length = labels_mask.sum(-1)
    if to_float64:
        reference_length = reference_length.to(torch.float64)
        log_probas = log_probas.to(torch.float64)
        label_log_probas = label_log_probas.to(torch.float64)
    # TODO: Make sure not `cum_log_probas`
    # [:, 1:] since we do not need it for the first token (BOS)
    token_av_logs = log_probas.mean(0)[:, 1:]
    # Transform log_probas to `cum_log_probas`
    label_cum_log_probas = label_log_probas.cumsum(-1)[:, :, :-1, None]
    log_probas[:, :, 1:] += label_cum_log_probas
    # ...
    label_cum_probas_logsumexp = torch.logsumexp(
        label_cum_log_probas, 0
    )  # batch_size x seq_len
    ep_posterior = (
        torch.logsumexp(log_probas[:, :, 1:], 0) - label_cum_probas_logsumexp
    ).exp()  # batch_size x seq_len x vocab_size
    ep_model = (
        log_probas[:, :, 1:] - label_cum_probas_logsumexp + log(log_probas.shape[0])
    ).exp()  # ens_size x batch_size x seq_len
    mean_entropy = (
        torch.tensor(entropy(ep_model.cpu().numpy(), axis=-1)).to(log_probas.device).mean(0).masked_fill(~labels_mask, 0)
    )  # batch_size x seq_len
    output["total_uncertainty"] = (
        torch.tensor(entropy(ep_posterior.cpu().numpy(), axis=-1)).to(log_probas.device).masked_fill(~labels_mask, 0).sum(-1) / reference_length
    )  # batch_size
    output["data_uncertainty"] = mean_entropy.sum(-1) / reference_length  # batch_size
    output["mutual_information"] = (
        output["total_uncertainty"] - output["data_uncertainty"]
    )

    output["epkl_total_uncertainty"] = (
        -(token_av_logs * ep_posterior).sum(-1).masked_fill(~labels_mask, 0).sum(-1)
        / reference_length
    )
    output["epkl"] = output["epkl_total_uncertainty"] - output["data_uncertainty"]
    output["rmi"] = output["epkl_total_uncertainty"] - output["total_uncertainty"]

    for k in entropy_topk:
        entropies_topk = torch.tensor(entropy(ep_posterior.topk(k, dim=-1).values.cpu().numpy(), axis=-1)).to(log_probas.device)
        output[f"entropy_topk{k}"] = (
            entropies_topk.masked_fill(~labels_mask, 0).sum(-1) / reference_length
        )

    del ep_posterior, ep_model, mean_entropy, label_cum_probas_logsumexp, token_av_logs
    torch.cuda.empty_cache()

    return output
