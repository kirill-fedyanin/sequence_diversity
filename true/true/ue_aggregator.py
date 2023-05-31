import logging
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t
from copy import deepcopy

from true.plot_metrics import plot_all
from true.ue_manager import UEManager
from true.ue_metrics import get_roc_auc_scores

log = logging.getLogger()


class UEAggregator:
    def __init__(
        self, managers: List[Union[UEManager, str, Path]] = None, device="cpu"
    ):
        """
        Aggregator for several managers (generally with different seeds)
        to ensure the robustness of results
        Args:
            managers: either a list of UEManager-s or paths to them
        """
        if managers is None:
            self.managers = []
        elif isinstance(managers[0], UEManager) or isinstance(managers[0], list):
            self.managers = managers
        elif isinstance(managers[0], (str, Path)):
            self.managers = []
            log.info("Loading UEManagers...")
            for path in managers:
                self.managers.append(UEManager.load(path, True, device))
                log.info(f"UEManagers {path} loaded.")
            log.info("Done with loading UEManagers.")
        else:
            raise TypeError(
                f"`managers` must contain either UEManagers or paths to them! Got type {type(managers[0])}"
            )

    def __call__(
        self,
        ue_methods=None,
        ue_metrics_names=None,
        seq2seq_metrics_names=None,
        call: bool = False,
        plot: bool = True,
        return_df: bool = True,
        return_figures: bool = False,
        show_figures: bool = False,
        colors=None,
        add_std_to_figures: bool = False,
        ood: bool = False,
        ood_labels=None,
        num_fp_digits_in_df: int = 3,
        **manager_kwargs,
    ) -> Union[list, pd.DataFrame, Tuple[pd.DataFrame, list]]:
        """
        :param ue_methods: UE methods to use. Default to the intersection of methods across all the managers
        :param ue_metrics_names: which UE metrics to use. Default: all (PRR, PRR_wo_oracle)
        :param seq2seq_metrics_names: which seq2seq metrics to use. Default: all (ROUGE-1, ROUGE-2, ROUGE-L, BLEU)
        :param call: whether to call each manager
        :param return_df: whether to return df with metrics
        :param return_figures: whether to return figures
        :param colors: colors to use for `plot_all` func
        :param add_std_to_figures: if True, then plots will have std shown
        :param ood_labels: true ood labels for plotting roc-auc (0 - ID, 1 - OOD)
        :param num_fp_digits_in_df: number of digits after floating point in df-s
        :param manager_kwargs: kwargs for UEManager.__call__ except for
        [`ue_names`, `seq2seq_metrics_names`, `plot`] args.
        :return: averaged scores & std
        """
        # Deal with OOD
        if ood and ood_labels is None:
            # Take into account case when different ue approaches where
            # calculated separately - outer list represents different seeds
            # and inner list represents differrent ue approaches for a given
            # seed
            if isinstance(self.managers[0], list):
                ood_labels = self.managers[0][0].ood_labels
            else:
                ood_labels = self.managers[0].ood_labels
        outputs, dfs = [], []
        mean_dict, lb_dict, ub_dict, std_dict = {}, {}, {}, {}

        # Need to modify this to accomodate both nested and flat lists of managers
        for mans in self.managers:
            df = []
            output = {}
            for appr_man in mans:
                if call:
                    outputs.append(
                        appr_man(
                            ue_names=ue_methods,
                            ats_metrics_names=ats_metrics_names,
                            plot=False,
                            return_df=False,
                            **manager_kwargs
                        )
                    )
                else:
                    output.update(appr_man.output[1])
                    # Temporary fix: for the already finished experiments (up to 16.01)
                    if ood:
                        roc_auc_scores = get_roc_auc_scores(
                            appr_man.ue_dict, ood_labels, list(appr_man.ue_dict.keys())
                        )
                        output.update(roc_auc_scores)
                    # Temporary fix end
                    outputs.append(output)
                df.append(appr_man.df)
            dfs.append(pd.concat(df, axis=1, ignore_index=True))

        for key in outputs[0]:
            values = [
                output.get(key, None)
                for output in outputs
                if output.get(key, None) is not None
            ]
            if key.startswith("data_ROC_AUС"):
                fpr_tpr_results, score = self._get_mean_tpr_fpr_roc_auc_score(values)
                mean_dict[key] = [fpr_tpr_results, score]
                # scores = [values[i][1] for i in range(len(self.managers))]
                # std_dict[key] = np.std(scores)
            else:
                mean, lb, ub, std = self._get_mean_lb_ub_std(values)
                mean_dict[key] = mean
                lb_dict[key] = lb
                ub_dict[key] = ub
                std_dict[key] = std
        self.mean_dict = mean_dict
        self.lb_dict = lb_dict
        self.ub_dict = ub_dict
        self.std_dict = std_dict
        self.outputs = outputs

        self._average_dfs(dfs, num_fp_digits_in_df=num_fp_digits_in_df)
        if ood:
            self._create_ood_df(
                mean_dict, std_dict, num_fp_digits_in_df=num_fp_digits_in_df
            )
        if plot:
            self.figs = plot_all(
                mean_dict=mean_dict,
                lb_dict=lb_dict,
                ub_dict=ub_dict,
                ue_methods=ue_methods,
                ue_metrics_names=ue_metrics_names,
                seq2seq_metrics_names=seq2seq_metrics_names,
                colors=colors,
                add_std_to_figures=add_std_to_figures,
                show_figures=show_figures,
            )
        output = tuple()
        if return_df:
            output += (self.df_style,)
        if plot and return_figures:
            output += (self.figs,)
        return output

    def __iter__(self):
        return self.managers

    def _average_dfs(self, dfs, num_fp_digits_in_df: int = 3):
        mean_df = sum(dfs) / len(dfs)
        std_df = dfs[0].copy(deep=True)
        std_df.iloc[:, :] = np.std(np.array(dfs).astype(float), axis=0)

        num_obs = len(dfs)
        mult_factor = t.ppf(0.975, num_obs) / num_obs ** (1 / 2)
        std_df *= mult_factor
        df = (
            mean_df.applymap(lambda x: str(round(x, num_fp_digits_in_df)))
            + " ± "
            + std_df.applymap(lambda x: str(round(x, num_fp_digits_in_df)))
        )

        self.df = df
        self.df_style = mean_df.style.background_gradient(cmap="Reds", axis=1)
        return self.df, self.df_style

    def _create_ood_df(self, mean_dict, std_dict, num_fp_digits_in_df: int = 3):
        # Save only ROC_AUC keys
        mean_df = pd.DataFrame(
            {key: value for key, value in mean_dict.items() if "ROC_AUC" in key},
            index=[0],
        )
        std_df = pd.DataFrame(
            {key: value for key, value in std_dict.items() if "ROC_AUC" in key},
            index=[0],
        )
        df = (
            mean_df.applymap(lambda x: str(round(x, num_fp_digits_in_df)))
            + " ± "
            + std_df.applymap(lambda x: str(round(x, num_fp_digits_in_df)))
        )
        self.df_ood = df
        self.df_ood_style = mean_df.style.background_gradient(cmap="Reds", axis=0)
        return self.df_ood, self.df_ood_style

    def _add_manager(self, manager: UEManager):
        self.managers.append(manager)

    @staticmethod
    def _get_mean_lb_ub_std(values):
        num_obs = len(values)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        conf_lev_val = t.ppf(0.975, num_obs)
        std_by_conf_val = std * conf_lev_val
        count_sqrt = num_obs ** (1 / 2)
        std = std_by_conf_val / count_sqrt
        # Calcaulate lower and upper bounds
        lb = mean - std
        ub = mean + std
        return mean, lb, ub, std

    @staticmethod
    def _get_mean_tpr_fpr_roc_auc_score(values):
        roc_auc_dicts = [x[0] for x in values]
        run_fprs = [np.array(list(x.keys())) for x in roc_auc_dicts]
        all_fprs = sorted({x for fpr in run_fprs for x in fpr})

        results = {}
        num_runs = len(run_fprs)
        cur_idx = dict.fromkeys(range(num_runs), 0)

        for fpr_step in all_fprs:
            tpr_step = 0
            for i, roc_auc_dict in enumerate(roc_auc_dicts):
                tpr = roc_auc_dict.get(fpr_step)
                if tpr is None:
                    # FPR value is between lv (lower value) and hv (higher values)
                    idx = cur_idx[i]
                    lv_and_hv = run_fprs[i][idx - 1 : idx + 1]
                    # Lower and higher value weights
                    weights = [1, 1]
                    weights_sum = 2
                    # Calculate initial threshold
                    threshold = (lv_and_hv[0] + lv_and_hv[1]) / weights_sum
                    while not np.allclose(threshold, fpr_step):
                        weights_sum += 1
                        if threshold > fpr_step:
                            weights[0] += 1
                        else:
                            weights[1] += 1
                        threshold = (
                            lv_and_hv[0] * weights[0] + lv_and_hv[1] * weights[1]
                        ) / weights_sum
                    tpr = (
                        roc_auc_dict[lv_and_hv[0]] * weights[0]
                        + roc_auc_dict[lv_and_hv[1]] * weights[1]
                    ) / weights_sum
                else:
                    cur_idx[i] += 1

                tpr_step += tpr
            results[fpr_step] = tpr_step / num_runs

        score = np.mean([x[1] for x in values])
        return results, score
