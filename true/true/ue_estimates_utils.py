from typing import List

import numpy as np
import torch
from datasets import load_metric
from tqdm import tqdm

from true.seq2seq_metrics import pair_bleu
from true.generate_utils import restore_token_level_data
from true.mc_utils import get_mc_output


def calculate_bleuvar_scores(hypotheses: List[List[str]]):
    """
    Given a list of generated texts, computes the pairwise BLEUvar
    between all text pairs. In addition, also finds the generation
    that has the smallest avg. BLEUvar score (most similar)
    with all other generations.

    :param hypotheses: list of lists with shape `num_obs x num_sum_per_obj`
    """
    bleu_vars, min_bleuvars, min_gen_idxs, min_gens = [], [], [], []
    for inst_hypotheses in tqdm(hypotheses, desc="Calculating BLEUVar scores..."):
        n = len(inst_hypotheses)
        bleu_scores = np.zeros((n, n), dtype=float)
        min_gen_idx = None
        min_bleuvar = float("inf")
        for j, dec_j in enumerate(inst_hypotheses):
            for k in range(j + 1, n):
                dec_k = inst_hypotheses[k]
                jk_bleu = pair_bleu(dec_j, dec_k)
                kj_bleu = pair_bleu(dec_k, dec_j)

                bleu_scores[j, k] = 1 - jk_bleu
                bleu_scores[k, j] = 1 - kj_bleu

            mu_bleuvar = np.sum(bleu_scores[j, :]) + np.sum(bleu_scores[:, j])
            if mu_bleuvar < min_bleuvar:
                min_bleuvar = mu_bleuvar
                min_gen_idx = j

        bleu_var = (bleu_scores**2).sum() / (n * (n - 1))
        min_gen = inst_hypotheses[min_gen_idx]

        bleu_vars.append(bleu_var)
        min_bleuvars.append(min_bleuvar)
        min_gen_idxs.append(min_gen_idx)
        min_gens.append(min_gen)

    return (
        np.array(bleu_vars),
        np.array(min_bleuvars),
        np.array(min_gen_idxs),
        np.array(min_gens),
    )


def calculate_bleuvar_with_deterministic_scores(
    stochastic_hypotheses: List[List[str]],
    deterministic_hypotheses: List[List[str]],
    symmetric: bool = True,
):
    """
    Given a list of generated texts, computes the BLEUvar
    between stochastic and deterministic results.
    """
    bleu_vars = []
    for stoch_s, det_s in tqdm(
        list(zip(stochastic_hypotheses, deterministic_hypotheses)),
        desc="Calculating BLEUVar Deterministic scores...",
    ):
        bleu_var = 0
        for j, stoch_hyp in enumerate(stoch_s):
            bleu = pair_bleu(det_s, stoch_hyp)
            if symmetric:
                rec = pair_bleu(stoch_hyp, det_s)
                bleu = 2 * bleu * rec / (bleu + rec + 1e-8)
            bleu_var += bleu

        bleu_vars.append(bleu_var)

    return -np.array(bleu_vars)


def calculate_metricvar_scores(
    hypotheses: List[List[str]], metric: str = "rouge1", **metric_kwargs
):
    """
    Given a list of generated texts, computes the pairwise Metric var (e.g.
    ROUGE-1-VAR)  between all text pairs. In addition, also finds the
    generation that has the smallest avg. Metric var score (most similar)
    with all other generations.
    """
    metric_name = metric
    if metric.startswith("rouge"):
        metric_name = "rouge"
        metric_kwargs["use_aggregator"] = False
        key = metric
    elif metric == "sacrebleu":
        metric_kwargs["smooth_method"] = "add-k"
        key = "score"
    else:
        raise NotImplementedError

    metric = load_metric(metric_name)
    metric_vars, min_metric_vars, min_gen_idxs, min_gens = [], [], [], []
    for inst_hypotheses in tqdm(hypotheses, desc="Calculating MetricVAR scores..."):
        n = len(inst_hypotheses)
        bleu_scores = np.zeros((n, n), dtype=float)
        min_gen_idx = None
        min_metric_var = float("inf")
        for j, dec_j in enumerate(inst_hypotheses):
            for k in range(j + 1, n):
                dec_k = inst_hypotheses[k]
                if metric_name == "rouge":
                    jk_bleu = metric.compute(
                        references=[dec_j], predictions=[dec_k], **metric_kwargs
                    )[key]
                    jk_bleu = kj_bleu = jk_bleu[0].fmeasure
                else:
                    jk_bleu = metric.compute(
                        references=[[dec_j]], predictions=[dec_k], **metric_kwargs
                    )[key]
                    kj_bleu = metric.compute(
                        references=[[dec_k]], predictions=[dec_j], **metric_kwargs
                    )[key]

                bleu_scores[j, k] = 1 - jk_bleu
                bleu_scores[k, j] = 1 - kj_bleu

            mu_bleuvar = np.sum(bleu_scores[j, :]) + np.sum(bleu_scores[:, j])
            if mu_bleuvar < min_metric_var:
                min_metric_var = mu_bleuvar
                min_gen_idx = j

        metric_var = (bleu_scores**2).sum() / (n * (n - 1))
        min_gen = inst_hypotheses[min_gen_idx]

        metric_vars.append(metric_var)
        min_metric_vars.append(min_metric_var)
        min_gen_idxs.append(min_gen_idx)
        min_gens.append(min_gen)

    return (
        np.array(metric_vars),
        np.array(min_metric_vars),
        np.array(min_gen_idxs),
        np.array(min_gens),
    )


def get_token_level_data(inference_output=None, **mc_output_kwargs):
    if inference_output is None:
        inference_output = get_mc_output(**mc_output_kwargs)
    num_obs = len(mc_output_kwargs["data"])
    num_beams = inference_output["scores"].shape[1]
    token_level_data = restore_token_level_data(inference_output, num_obs, num_beams)
    return token_level_data


def make_tensor_set_device(tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Tensor(tensor).to(device)
