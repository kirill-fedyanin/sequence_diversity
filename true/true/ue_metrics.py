from typing import Dict, Union, List

import numpy as np
import sklearn.metrics as sk_metrics

from true.default_values import DEFAULT_UE_METRICS


def calculate_prediction_rejection_area(ue, metrics, use_oracle=True):
    """
    PRR metric
    """
    num_obs = len(metrics)
    assert (
        len(ue) == num_obs or len(ue.shape) == len(metrics) == 1
    ), f"UE and metrics must be of the same length, but got values {len(ue)}, {num_obs}!"
    # Sort in ascending order: the least uncertain come first
    ue_argsort = np.argsort(ue)
    sorted_metrics = np.array(metrics)[ue_argsort]
    # Since we want all plots to coincide when all the data is discarded
    cumsum = np.cumsum(sorted_metrics)
    if use_oracle:
        scores = (cumsum[::-1] + np.arange(num_obs)) / num_obs
    else:
        scores = (cumsum / np.arange(1, num_obs + 1))[::-1]

    prr_score = np.sum(scores) / num_obs
    scores = np.append(scores, 1)

    return prr_score, scores


def calculate_rcc_auc(ue, metrics, normalize=True):
    # risk-coverage curve's area under curve
    conf = -ue
    risk = 1 - metrics
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = np.cumsum([x[1] for x in cr_pair])
    if normalize:
        cumulative_risk = cumulative_risk / np.arange(1, n + 1)
    rcc_auc = cumulative_risk.sum()

    return rcc_auc, cumulative_risk


def get_random_scores(function, metrics, num_iter=10, seed=42, **kwargs):
    np.random.seed(seed)
    rand_scores = np.arange(len(metrics))

    value, scores = [], []
    for i in range(num_iter):
        np.random.shuffle(rand_scores)
        rand_val, rand_metric_scores = function(rand_scores, metrics, **kwargs)
        value.append(rand_val)
        scores.append(rand_metric_scores)
    return np.mean(value), np.mean(scores, 0)


def get_oracle_scores(function, metrics, **kwargs):
    return function(-metrics, metrics, **kwargs)


def get_all_scores(
    ue: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray],
    seq2seq_metrics: Dict[str, Dict[str, np.ndarray]],
    add_random_and_oracle=False,
    ue_methods=None,
    ue_metrics_names=None,
    ood_labels: Union[List[int], np.ndarray] = None,
    **random_kwargs,
) -> dict:
    if ue_metrics_names is None:
        ue_metrics_names = DEFAULT_UE_METRICS
    if ue_methods is None:
        assert isinstance(
            ue, dict
        ), f"Either `ue_methods` should be provided or `ue` must be a dict (got {type(ue)})!"
        ue_methods = ue.keys()
    if (len(ue) != len(ue_methods)) and (len(ue.shape) == len(ue_methods) == 1):
        ue = [ue]
    elif isinstance(ue, dict):
        ue = list(ue.values())

    scores_dict = {}

    FUNCS = (
        calculate_prediction_rejection_area,
        calculate_rcc_auc,
    )
    KWARGS = ({"use_oracle": True}, {"normalize": True})  # , {"use_oracle": False}

    # Get list of seq2seq metrics names
    seq2seq_metric_names = seq2seq_metrics[next(iter(seq2seq_metrics.keys()))].keys()
    for seq2seq_metric_name in seq2seq_metric_names:
        metrics = {
            key: value[seq2seq_metric_name] for key, value in seq2seq_metrics.items()
        }
        # Metrics may contain NaN values
        metric_is_nan = {key: np.isnan(value) for key, value in metrics.items()}
        metrics = {
            key: np.array(value)[~metric_is_nan[key]] for key, value in metrics.items()
        }
        # ue_seq2seq_metrics = [x.ravel()[~metric_is_nan] for x in ue]
        ue_seq2seq_metrics = [x.ravel() for x in ue]

        for method_ue, method_name in zip(ue_seq2seq_metrics, ue_methods):
            key = (
                "ensemble"
                if (method_name.startswith("EP-") or method_name.startswith("PE-"))
                else "inference"
            )
            metric = metrics[key]
            method_ue = method_ue[~metric_is_nan[key]]
            for func, kw, metric_name in zip(FUNCS, KWARGS, ue_metrics_names):
                score, scores = func(method_ue, metric, **kw)
                scores_dict[
                    f"{seq2seq_metric_name}_{method_name}_{metric_name}_score"
                ] = score
                scores_dict[
                    f"{seq2seq_metric_name}_{method_name}_{metric_name}_scores"
                ] = scores

        if add_random_and_oracle:
            metric = (
                metrics["inference"] if "inference" in metrics else metrics["ensemble"]
            )
            for func, kw, metric_name in zip(FUNCS, KWARGS, ue_metrics_names):
                rand_score, rand_scores = get_random_scores(
                    func, metric, **kw, **random_kwargs
                )
                oracle_score, oracle_scores = get_oracle_scores(func, metric, **kw)
                scores_dict[
                    f"{seq2seq_metric_name}_rand_{metric_name}_score"
                ] = rand_score
                scores_dict[
                    f"{seq2seq_metric_name}_rand_{metric_name}_scores"
                ] = rand_scores
                scores_dict[
                    f"{seq2seq_metric_name}_oracle_{metric_name}_score"
                ] = oracle_score
                scores_dict[
                    f"{seq2seq_metric_name}_oracle_{metric_name}_scores"
                ] = oracle_scores

                for method_name in ue_methods:
                    scores_dict[
                        f"{seq2seq_metric_name}_{method_name}_{metric_name}_target_score"
                    ] = get_target_score(
                        scores_dict, method_name, seq2seq_metric_name, metric_name
                    )
    if ood_labels is not None:
        roc_auc_scores = get_roc_auc_scores(ue, ood_labels, ue_methods)
        scores_dict.update(roc_auc_scores)

    return scores_dict


def get_target_score(
    scores_dict, method, seq2seq_metric_name="rouge1", metric="prr"
) -> float:
    metric_score = scores_dict[f"{seq2seq_metric_name}_{method}_{metric}_score"]
    oracle_score = scores_dict[f"{seq2seq_metric_name}_oracle_{metric}_score"]
    rand_score = scores_dict[f"{seq2seq_metric_name}_rand_{metric}_score"]
    if oracle_score == rand_score:
        return 0
    return (metric_score - rand_score) / (oracle_score - rand_score)


def get_roc_auc_scores(
    ue: Dict[str, np.ndarray],
    labels: Union[List[int], np.ndarray],
    ue_methods: Union[List[str], np.ndarray] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    if ue_methods is None:
        ue_methods = ue.keys()
    roc_auc_scores = {}
    for method in ue_methods:
        scores = ue[method]
        scores[np.isnan(scores)] = 0
        fpr, tpr, _ = sk_metrics.roc_curve(labels, scores)
        score = sk_metrics.auc(fpr, tpr)
        roc_auc_scores[f"ROC_AUC_{method}"] = score
        # Need to store only unique values in FPR to make it y(x)
        fpr_to_tpr_dict = dict(zip(fpr, tpr))
        roc_auc_scores[f"data_ROC_AUС_{method}"] = [fpr_to_tpr_dict, score]
    return roc_auc_scores
