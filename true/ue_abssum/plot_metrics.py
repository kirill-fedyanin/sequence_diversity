import logging
from typing import Tuple, Dict, Union, List

import numpy as np
from plotly import graph_objects as go

from ue_abssum.default_values import DEFAULT_UE_METRICS, COLORS
from ue_abssum.ue_metrics import get_all_scores, get_target_score
from ue_abssum.utils import get_seq2seq_metrics_names, make_method_upper

log = logging.getLogger()


def compare_with_random_and_oracle(
    ue=None,
    seq2seq_metrics: Dict[str, Dict[str, Union[List[float], np.ndarray]]] = None,
    scores_dict=None,
    ue_methods=None,
    ue_metrics_names=None,
    **random_kwargs,
) -> Tuple[list, dict]:
    if ue_metrics_names is None:
        ue_metrics_names = DEFAULT_UE_METRICS
    if ue_methods is None:
        assert isinstance(
            ue, dict
        ), f"Either `ue_methods` should be provided or `ue` must be a dict (got {type(ue)})!"
        ue_methods = ue.keys()

    if scores_dict is None:
        scores_dict = get_all_scores(
            ue,
            seq2seq_metrics=seq2seq_metrics,
            add_random_and_oracle=True,
            ue_methods=ue_methods,
            **random_kwargs,
        )

    figs = []
    # Need only names and lengths here
    key = "inference" if "inference" in seq2seq_metrics else "ensemble"
    for seq2seq_metric_name, seq2seq_metric in seq2seq_metrics[key].items():
        log.info("=" * 15 + f"Seq2Seq Metric {seq2seq_metric_name}" + "=" * 15 + "\n")

        for metric in ue_metrics_names:
            rand_scores = scores_dict[f"{seq2seq_metric_name}_rand_{metric}_scores"]
            oracle_scores = scores_dict[f"{seq2seq_metric_name}_oracle_{metric}_scores"]

            x_axis = np.arange(len(seq2seq_metric) + 1) / (len(seq2seq_metric) + 1)

            fig = go.Figure(
                layout=dict(
                    # height=400, width=550, title=metric
                    height=400,
                    width=700,
                    title=metric + ", " + seq2seq_metric_name,
                    margin=dict(l=0, r=0, t=30, b=10),
                )
            )
            for method in ue_methods:
                score = get_target_score(
                    scores_dict, method, seq2seq_metric_name, metric
                )
                if method.startswith("EP-") or method.startswith("PE-"):
                    len_metric_obs_p_1 = (
                        len(seq2seq_metrics["ensemble"][seq2seq_metric_name]) + 1
                    )
                    x_axis = np.arange(len_metric_obs_p_1) / len_metric_obs_p_1
                log.info(f"{seq2seq_metric_name} {metric} {method} score: {score:.4f}")
                fig.add_scatter(
                    x=x_axis,
                    y=scores_dict[f"{seq2seq_metric_name}_{method}_{metric}_scores"],
                    name=method.upper() + "; " + str(round(score, 3)),
                )
            fig.add_scatter(x=x_axis, y=rand_scores, name=f"Random")
            fig.add_scatter(x=x_axis, y=oracle_scores, name=f"Oracle")
            fig.show()
            figs.append(fig)

    return figs, scores_dict


def plot_all(
    mean_dict,
    lb_dict,
    ub_dict,
    ue_methods=None,
    ue_metrics_names=None,
    seq2seq_metrics_names=None,
    colors=None,
    add_std_to_figures: bool = False,
    show_figures: bool = False,
):
    if ue_methods is None:
        ue_methods = [
            x
            for x in set((x.split("_")[1] for x in mean_dict.keys()))
            if x not in ["rand", "oracle", "prr", "ROC", "AUC"]
        ]
    if ue_metrics_names is None:
        ue_metrics_names = DEFAULT_UE_METRICS
    if seq2seq_metrics_names is None:
        seq2seq_metrics_names = get_seq2seq_metrics_names(mean_dict)

    if colors is None:
        colors = COLORS
    sleeve_colors = list(map(_make_sleeve_color, colors))
    figs = []

    for seq2seq_metric_name in seq2seq_metrics_names:
        for metric in ue_metrics_names:
            rand_scores = mean_dict[f"{seq2seq_metric_name}_rand_{metric}_scores"]
            oracle_scores = mean_dict[f"{seq2seq_metric_name}_oracle_{metric}_scores"]

            x_axis = np.arange(len(rand_scores) + 1) / (len(rand_scores) + 1)

            fig = go.Figure(
                layout=dict(
                    # height=400, width=550, title=metric
                    height=400,
                    width=700,
                    title=metric + ", " + seq2seq_metric_name,
                    margin=dict(l=0, r=0, t=30, b=10),
                )
            )
            for i, method in enumerate(ue_methods):
                method_upper = make_method_upper(method)
                method = method_upper + method[len(method_upper) :]
                scores_key = f"{seq2seq_metric_name}_{method}_{metric}_scores"

                target_score = mean_dict[
                    f"{seq2seq_metric_name}_{method}_{metric}_target_score"
                ]
                log.info(f"{metric} {method} score: {target_score:.4f}")
                fig.add_scatter(
                    x=x_axis,
                    y=mean_dict[scores_key],
                    name=f"{method}, {target_score:.3f}",
                )
                if add_std_to_figures:
                    # Plot confidence interval
                    # Upper bound
                    fig.add_scatter(
                        x=x_axis,
                        y=ub_dict[scores_key],
                        name=method + "_upper_bound",
                        marker=dict(color="#444", size=0),
                        line=dict(width=0),
                        showlegend=False,
                        mode="lines",
                    )
                    # Lower bound
                    fig.add_scatter(
                        x=x_axis,
                        y=lb_dict[scores_key],
                        name=method + "_lower_bound",
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        fillcolor=sleeve_colors[i],
                        fill="tonexty",
                        showlegend=False,
                        mode="lines",
                    )

            fig.add_scatter(x=x_axis, y=rand_scores, name=f"Random")
            fig.add_scatter(x=x_axis, y=oracle_scores, name=f"Oracle")
            if show_figures:
                fig.show()
            figs.append(fig)
    if any((key.startswith("data_ROC_AUС") for key in mean_dict.keys())):
        fig = plot_roc_auc(
            ue_dict=mean_dict, ue_methods=ue_methods, show_figure=show_figures
        )
        figs.append(fig)

    return figs


def plot_roc_auc(
    ue_dict: Dict[str, np.ndarray],
    ue_methods: Union[List[str], np.ndarray] = None,
    show_figure: bool = False,
):
    if ue_methods is None:
        ue_methods = ue_dict.keys()
    fig = go.Figure(
        layout=dict(
            height=400,
            width=700,
            title="ROC-AUC",
            margin=dict(l=0, r=0, t=30, b=10),
        )
    )
    for method in ue_methods:
        results, score = ue_dict[f"data_ROC_AUС_{method}"]
        fig.add_scatter(
            x=[0] + list(results.keys()),
            y=[0] + list(results.values()),
            name=f"{method}, {score:.3f}",
        )
    if show_figure:
        fig.show()
    return fig


def _make_sleeve_color(color):
    return "rgba" + color[3:-1] + ", 0.2)"
