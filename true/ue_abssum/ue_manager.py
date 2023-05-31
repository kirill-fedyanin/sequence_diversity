import logging
import pickle
from functools import partial
from typing import List, Dict, Union, Tuple
from collections import defaultdict

import datasets
import dill
import numpy as np
import pandas as pd
import torch.cuda
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from peft import PeftModel, PeftConfig
from ue_abssum.seq2seq_metrics import (
    calculate_rouge,
    calculate_bleu,
    calculate_summac,
    calculate_bartscore,
    calculate_top_1_acc,
    calculate_top_all_acc,
    calculate_mrr,
    prepare_predictions_and_labels_for_rouge,
)

from ue_abssum.default_values import (
    DEFAULT_SEQ2SEQ_METRICS,
    DEFAULT_SEQ2SEQ_BASE_METRICS,
    DEFAULT_UE_METHODS,
    REV_OUTPUT_TYPE,
)
from ue_abssum.generate import generate
from ue_abssum.plot_metrics import compare_with_random_and_oracle
from ue_abssum.tokenize_data import tokenize_data
from ue_abssum.ue_estimates import StrategyManager
from ue_abssum.ue_generator import UeGenerator
from ue_abssum.ue_metrics import get_all_scores
from ue_abssum.utils import (
    get_default_ue_and_seq2seq_metrics_names,
    make_method_upper,
    _set_device,
)

log = logging.getLogger()

STRATEGIES = {
    "NSP": StrategyManager.nsp,
    "NSP-B": StrategyManager.nsp_biased,
    "USP": StrategyManager.usp,
    "MSP": StrategyManager.msp,
    "ENTROPY": StrategyManager.entropy,
    "ENTROPY5": StrategyManager.entropy_top5,
    "ENTROPY10": StrategyManager.entropy_top10,
    "ENTROPY15": StrategyManager.entropy_top15,
    "ENTROPY-S": StrategyManager.entropy_s,
    "ENTROPY-S-U": StrategyManager.entropy_s_u,
    "ENSP": StrategyManager.ensp,
    "MNSP": StrategyManager.mnsp,
    "ENSV": StrategyManager.ensv,
    "EDSLV": StrategyManager.edslv,
    "EDSSV": StrategyManager.edssv,
    "EDSL": StrategyManager.edsl,
    "EDSS": StrategyManager.edss,
    "BALD": StrategyManager.bald,
    "AVGLOSS": StrategyManager.avgloss,
    "ELOSS": StrategyManager.eloss,
    "BLEUVAR": StrategyManager.bleuvar,
    "BLEUVARDET": StrategyManager.bleuvar_deterministic,
    "ROUGE1VAR": partial(StrategyManager.metricvar, metric="rouge1"),
    "ROUGE2VAR": partial(StrategyManager.metricvar, metric="rouge2"),
    "ROUGELVAR": partial(StrategyManager.metricvar, metric="rougeL"),
    "SACREBLEUVAR": partial(StrategyManager.metricvar, metric="sacrebleu"),
    "MD": StrategyManager.mahalanobis_distance,
    "NUQ": StrategyManager.nuq,
    "DDU": StrategyManager.ddu,
    "RDE": StrategyManager.rde,
    "MD-ENCODER": StrategyManager.mahalanobis_distance,
    "NUQ-ENCODER": StrategyManager.nuq,
    "DDU-ENCODER": StrategyManager.ddu,
    "RDE-ENCODER": StrategyManager.rde,
    "MD-DECODER": StrategyManager.mahalanobis_distance,
    "NUQ-DECODER": StrategyManager.nuq,
    "DDU-DECODER": StrategyManager.ddu,
    "RDE-DECODER": StrategyManager.rde,
    "ENC-DIFF": StrategyManager.enc_diff,
    "EMB-MLP": StrategyManager.emb_mlp,
    "DE-NSP": partial(StrategyManager().deep_ensemble, strategy="ensp"),
    "DE-SP": partial(StrategyManager().deep_ensemble, strategy="esp"),
    "DE-BLEUVAR": partial(StrategyManager().deep_ensemble, strategy="bleuvar"),
    "DE-ROUGE1VAR": partial(
        StrategyManager().deep_ensemble,
        strategy="bleuvar",
        strategy_kwargs={"metric": "rouge1"},
    ),
    "PE-TOK": StrategyManager.pe_token_unc,
    "PE-SEQ": StrategyManager.pe_seq_unc,
    "EP-TOK": StrategyManager.ep_token_unc,
    "EP-SEQ": StrategyManager.ep_seq_unc,
    "SEP-SEQ": partial(StrategyManager.ep_seq_unc, is_based_on_single_output=True),
    "SEP-TOK": partial(StrategyManager.ep_token_unc, is_based_on_single_output=True),
}


class UEManager:
    def __init__(
        self,
        ue_methods,
        model=None,
        peft_id=None,
        tokenizer=None,
        configs_path=None,
        data=None,
        data_config=None,
        ood_labels: Union[List[int], np.ndarray] = None,
        seed: int = 42,
        output_dict=None,
        ue_dict=None,
        performance_dict=None,
        seq2seq_metrics=None,
        output=None,
        df: pd.DataFrame = None,
        tokenize: bool = True,
        raw_data=None,  # ignored if `tokenize`
        device: str = "0",
    ):
        self.ue_methods = ue_methods if ue_methods is not None else DEFAULT_UE_METHODS
        device = _set_device(device)
        self.model_path = model
        self.tokenizer_path = tokenizer
        if model is not None:
            self.model, self.tokenizer = self._static_load_model_and_tokenizer(
                model, peft_id, tokenizer, device
            )
        else:
            self.model, self.tokenizer = None, None
        self.seed = seed

        self.ue_gen = UeGenerator(configs_path=configs_path,
                                  output_dict=output_dict,
                                  seed=seed)
        self.ue_dict = ue_dict if ue_dict is not None else {}
        self.performance_dict = performance_dict
        self.seq2seq_metrics = seq2seq_metrics if seq2seq_metrics is not None else {}
        self.output = output if output is not None else {}
        self.df = df
        self.performance_dict = None

        if data_config is None:
            data_config = DictConfig({"text_name": "document", "label_name": "summary"})
        self._data_text_name = getattr(data_config, "text_name", "document")
        self._data_label_name = getattr(data_config, "label_name", "summary")
        if tokenize and data is not None:
            self.raw_data = data
            self.data = tokenize_data(
                data, self.tokenizer, self._data_text_name, self._data_label_name
            )
        else:
            self.data = data
            if raw_data is not None:
                self.raw_data = raw_data
            else:
                self.raw_data = data
        self.ood_labels = ood_labels
        self.data_config = data_config

    def __call__(
        self,
        data=None,
        ue_methods: Union[List[str], Tuple[str], np.ndarray] = None,
        update_ue_gen: bool = False,
        update_ue_dict: bool = False,
        data_config=None,
        update_metrics: bool = False,
        labels: Union[List[int], np.ndarray] = None,
        ood_labels: Union[List[int], np.ndarray] = None,
        update_output: bool = False,
        ue_metrics_names=None,
        seq2seq_metrics_names=None,
        plot: bool = True,
        return_df: bool = True,
        return_figures: bool = False,
        update_all: bool = False,
        **kwargs,
    ):
        """

        Args:
            data:
            ue_methods: methods for UE that we will use (e.g. `[NSP, PE-SEQ]` etc.)
            update_ue_gen:  whether to update the dict with the generated data for UE estimates (e.g. ensemble inference for PE/EP - seq/tok)
            update_ue_dict: whether to recalculate the UE estimates (e.g. NSP etc.)
            data_config:
            update_metrics: whether to update the UE metrics (e.g. ROUGEs etc.)
            labels:
            ood_labels:
            update_output:
            ue_metrics_names:
            seq2seq_metrics_names:
            plot:
            return_df:
            return_figures:
            update_all:
            **kwargs:

        Returns:

        """
        if update_all:
            update_ue_gen = update_metrics = update_output = update_output_dict = True
        if (data is not None) and ("input_ids" not in data.features):
            if data_config is None:
                data_config = self.data_config
            label_name = data_config["label_name"]
            data = tokenize_data(
                data, self.tokenizer, data_config["text_name"], label_name
            )
            if labels is None:
                labels = data[label_name]
        elif data is None and labels is None:
            labels = self.raw_data[self._data_label_name]

        if update_ue_gen or update_ue_dict or len(self.ue_dict) == 0:
            ue_dict = self.get_ue_dict(
                data=data,
                ue_methods=ue_methods,
                update_ue_gen=update_ue_gen,
                update_ue_dict=update_ue_dict,
                data_config=data_config,
                **kwargs,
            )
        else:
            ue_dict = self.ue_dict

        if update_metrics or len(self.seq2seq_metrics) == 0:
            seq2seq_metrics, _ = self.calculate_seq2seq_metrics(
                inference_data=data,
                labels=labels,
                data_config=data_config,
                ood_labels=ood_labels,
                calculate_performance=True,
                **kwargs,
            )
        else:
            seq2seq_metrics = self.seq2seq_metrics

        if update_output or len(self.output) == 0:
            output = self.calculate_scores(
                seq2seq_metrics=seq2seq_metrics,
                ue_dict=ue_dict,
                ue_methods=ue_methods,
                ue_metrics_names=ue_metrics_names,
                seq2seq_metrics_names=seq2seq_metrics_names,
                ood_labels=ood_labels,
                plot=plot,
                return_df=return_df,
                return_figures=return_figures,
            )
        else:
            output = self.output
        if plot and isinstance(output, tuple) and isinstance(output[0], list):
            for fig in output[0]:
                fig.show()

        self.collect_dropout_results(data, data_config, labels, **kwargs)

        return output

    def collect_dropout_results(self, data, data_config, labels, **kwargs):
        dropout_rates = kwargs.get('dropout_rates', [])
        inference_seed = kwargs.get('inference_seed', None) 
        _type = self.model_path.split('/')[-1]
        _, labels = self._get_texts_and_labels(self.data, self.data_config, labels)
        if len(dropout_rates) > 0:
            accs = {}
            results = {}
            for p in dropout_rates:
                print(f"Calculate top 1 acc for {p}")
                p_results = self.ue_gen._output_dict[f'inference_{p}']['wd_ids']
                acc = np.mean(calculate_top_1_acc(p_results, labels)['Top 1 Acc'])
                accs[p] = acc
                results[p] = p_results
        if inference_seed is not None:
            print("Saving dropout results...")
            with open(f'{_type}_{inference_seed}_varying_dropout_accs.pkl', 'wb') as handle:
                pickle.dump((accs, results), handle)

    def get_ue_dict(
        self,
        data=None,
        ue_methods=None,
        update_ue_gen=False,
        update_ue_dict=False,
        data_config=None,
        **kwargs,
    ):
        if data is None:
            data = self.data
        elif "input_ids" not in data.features:
            # Check whether is necessary
            raw_data = data
            data = tokenize_data(
                data, self.tokenizer, self._data_text_name, self._data_label_name
            )
        if data_config is None:
            data_config = self.data_config
        if ue_methods is None:
            ue_methods = self.ue_methods
        elif isinstance(ue_methods, str):
            ue_methods = [ue_methods]

        ue_dict = {} if update_ue_dict else self.ue_dict
        generate_kwargs = dict(
            model=self.model,
            data=data,
            tokenizer=self.tokenizer,
            generation_max_length=kwargs.get("generation_max_length", None),
            batch_size=kwargs.get("batch_size", None),
            num_return_sequences=kwargs.get("num_return_sequences", 4),
            data_config=data_config,
            is_tokenized=("input_ids" in data.features),
            to_numpy=True,
            aggregate_sequences_scores=True,
            t_for_weights=kwargs.get("t_for_weights", 1.0),
            kg_id_mapping=kwargs.get("kg_id_mapping", None),
            prefix_allowed_tokens_fn=kwargs.get('prefix_allowed_tokens_fn', None),
            dropout_rates=kwargs.get('dropout_rates', []),
            inference_seed=kwargs.get('inference_seed', None),
            num_beam_groups=kwargs.get('num_beam_groups', None),
            diversity_penalty=kwargs.get('diversity_penalty', None),
            repetition_penalty=kwargs.get('repetition_penalty', None),
            do_sample=kwargs.get('do_sample', None),
            top_k=kwargs.get('top_k', None),
            top_p=kwargs.get('top_p', None)
        )

        self.ue_gen._create_output_dict(
            update_ue_gen, generate_kwargs, ue_methods, **kwargs
        )
        for method in tqdm(ue_methods, desc="UE scores calculated..."):
            # If the required inference was successfully generated
            try:
                (
                    method_name_upper,
                    strategy_kwargs,
                ) = self._get_method_name_and_strategy_kwargs(
                    method=method, kwargs=kwargs
                )
            except Exception as e:
                log.info(f"Could not load data for method {method}: {e}")
                continue
            log.info(f"Method {method}, strategy {method_name_upper}:")
            scores = STRATEGIES[method_name_upper](**strategy_kwargs, ue_dict=ue_dict)
            if isinstance(scores, dict):
                ue_dict.update(scores)
            else:
                ue_dict[method] = scores
            self.ue_dict = ue_dict
        return ue_dict

    def calculate_seq2seq_metrics(
        self,
        inference_output: Dict[str, Dict[str, List[str]]] = None,
        inference_data=None,
        metrics_to_use=None,
        labels=None,
        data_config=None,
        ood_labels=None,
        calculate_performance: bool = True,  # whether to calculate seq metrics (e.g. ensemble SummaC-score: 0.25)
        **kwargs,
    ) -> Union[Dict[str, Dict[str, Union[np.ndarray, List[float]]]], Tuple[dict, dict]]:
        if inference_output is None:
            inference_output = {"inference": self.ue_gen._output_dict["inference"]}
            if "ensemble" in self.ue_gen._output_dict.keys():
                inference_output["ensemble"] = self.ue_gen._output_dict["ensemble"]["inference_output"]
        if inference_data is None:
            inference_data = self.data
        if data_config is None:
            data_config = self.data_config
        # if ood_labels is None:
        #     ood_labels = self.ood_labels
        if metrics_to_use is None:
            metrics_to_use = DEFAULT_SEQ2SEQ_BASE_METRICS
        else:
            metrics_to_use = [x.lower() for x in metrics_to_use]

        log.info("Calculating metrics...")
        predictions = {
            key: value["hypotheses"] for key, value in inference_output.items()
        }
        prediction_wd_ids = {
            key: value["wd_ids"] for key, value in inference_output.items() if "wd_ids" in value
        }
        texts, labels = self._get_texts_and_labels(inference_data, data_config, labels)
        seq2seq_metrics = {key: {} for key in inference_output.keys()}
        for key in seq2seq_metrics:
            predictions_key = predictions[key]
            if key in prediction_wd_ids:
                prediction_wd_ids_key = prediction_wd_ids[key]
            if "rouge" in metrics_to_use:
                # Prepare
                (
                    predictions_for_rouge,
                    labels_for_rouge,
                    is_zeroword,
                    is_uniword,
                ) = prepare_predictions_and_labels_for_rouge(predictions_key, labels)
                seq2seq_metrics[key].update(
                    calculate_rouge(
                        predictions_for_rouge, labels_for_rouge, is_zeroword, is_uniword
                    )
                )
            if "bleu" in metrics_to_use:
                seq2seq_metrics[key].update(calculate_bleu(predictions_key, labels))
            if "bartscore" in metrics_to_use:
                bartscore_kwargs = kwargs.get("bartscore", {})
                seq2seq_metrics[key].update(
                    calculate_bartscore(
                        predictions_key, labels, texts, **bartscore_kwargs
                    )
                )
            if "top_1_acc" in metrics_to_use:
                seq2seq_metrics[key].update(
                    calculate_top_1_acc(prediction_wd_ids_key, labels)
                )
            if "top_all_acc" in metrics_to_use:
                seq2seq_metrics[key].update(
                    calculate_top_all_acc(prediction_wd_ids_key, labels)
                )
            if "mrr" in metrics_to_use:
                seq2seq_metrics[key].update(
                    calculate_mrr(prediction_wd_ids_key, labels)
                )

            # TODO: probably remove, do not need accuracy here
            # if "accuracy" in metrics_to_use:
            #     if ood_labels is None:
            #         log.warning(
            #             "Accuracy will not be calculated since ood_labels are missing."
            #         )
            #     else:
            #         seq2seq_metrics[key].update({"Accuracy": 1 - ood_labels})

            seq2seq_metrics[key] = {
                k: np.array(v) for k, v in seq2seq_metrics[key].items()
            }
        log.info("Done.")
        self.seq2seq_metrics = seq2seq_metrics
        if not calculate_performance:
            return seq2seq_metrics

        log.info("Calculating performance metrics...")
        results = seq2seq_metrics.copy()
        for key in results:
            predictions_key = predictions[key]
            if "bartscore" in metrics_to_use:
                bartscore_kwargs = kwargs.get("bartscore", {})
                results[key].update(
                    calculate_bartscore(
                        predictions_key,
                        labels,
                        texts,
                        directions=("sh",),
                        **bartscore_kwargs,
                    )
                )
            if "summac" in metrics_to_use:
                results[key].update(
                    calculate_summac(
                        predictions_key,
                        labels,
                        texts,
                    )
                )
            results[key] = {k: np.nanmean(v) for k, v in results[key].items()}
            log.info(f"Generation type: {key}")
            for metric, score in results[key].items():
                log.info(f"{metric}: {score:.4f}")
        self.performance_dict = results
        log.info("Done...")
        return seq2seq_metrics, results


    def calculate_scores(
        self,
        seq2seq_metrics: Dict[str, Dict[str, np.ndarray]] = None,
        ue_dict: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        ue_methods: List[str] = None,
        ue_metrics_names: List[str] = None,
        seq2seq_metrics_names: List[str] = None,
        ood_labels: Union[List[int], np.ndarray] = None,
        plot: bool = True,
        return_df: bool = True,
        return_figures: bool = False,
        **calculate_metrics_kwargs,
    ):
        if seq2seq_metrics is None:
            if self.seq2seq_metrics is None:
                seq2seq_metrics = self.calculate_seq2seq_metrics(
                    **calculate_metrics_kwargs
                )
            else:
                seq2seq_metrics = self.seq2seq_metrics
        if seq2seq_metrics_names is None:
            seq2seq_metrics_names = DEFAULT_SEQ2SEQ_METRICS
        if ue_dict is None:
            ue_dict = self.ue_dict
        if ue_methods is None:
            ue_methods = ue_dict.keys()

        seq2seq_metrics = {
            key: {k: v for k, v in value.items() if k in seq2seq_metrics_names}
            for key, value in seq2seq_metrics.items()
        }

        if plot:
            output = compare_with_random_and_oracle(
                [ue_dict[x] for x in ue_methods],
                seq2seq_metrics,
                ue_methods=ue_methods,
                ue_metrics_names=ue_metrics_names,
            )
        else:
            output = get_all_scores(
                [ue_dict[x] for x in ue_methods],
                seq2seq_metrics=seq2seq_metrics,
                ue_methods=ue_methods,
                ue_metrics_names=ue_metrics_names,
                add_random_and_oracle=True,
                ood_labels=ood_labels,
            )
        self.output = output

        self._create_df_from_output(ue_methods, ue_metrics_names, seq2seq_metrics_names)
        if return_df:
            if plot and return_figures:
                output = output + (self.df_style,)
            elif plot:
                output = (output[1], self.df_style)
            else:
                output = (output, self.df_style)

        return output

    def generate(
        self,
        data: datasets.Dataset = None,
        is_tokenized: bool = False,
        data_config: dict = None,
        to_numpy: bool = False,
        to_eval_mode: bool = True,
        generation_max_length: int = None,
        num_return_sequences: int = 1,
        batch_size: int = None,
        **kwargs,
    ):
        if data is None:
            data = self.data
        if data_config is None:
            data_config = self.data_config
        return generate(
            model=self.model,
            data=data,
            tokenizer=self.tokenizer,
            is_tokenized=is_tokenized,
            generation_max_length=generation_max_length,
            num_return_sequences=num_return_sequences,
            data_config=data_config,
            to_numpy=to_numpy,
            to_eval_mode=to_eval_mode,
            batch_size=batch_size,
            **kwargs,
        )

    def _create_df_from_output(
        self, ue_methods, ue_metrics_names=None, seq2seq_metrics_names=None
    ):
        pd.set_option("display.max_rows", None)

        (
            ue_metrics_names,
            seq2seq_metrics_names,
        ) = get_default_ue_and_seq2seq_metrics_names(
            ue_metrics_names, seq2seq_metrics_names
        )
        scores_dict = (
            self.output
            if not isinstance(self.output, tuple)
            or not isinstance(self.output[0], list)
            else self.output[1]
        )
        df = pd.DataFrame(columns=["seq2seq_metric", "ue_metric", "ue_method", "score"])
        for seq2seq_metric in seq2seq_metrics_names:
            for metric in ue_metrics_names:
                for ue_method in ue_methods:
                    score_name = f"{seq2seq_metric}_{ue_method}_{metric}_target_score"
                    if score_name in scores_dict:
                        df.loc[len(df)] = [seq2seq_metric, metric, ue_method] + [
                            scores_dict[score_name]
                        ]
        self.df = df.pivot(
            index=["seq2seq_metric", "ue_metric"], columns="ue_method", values="score"
        )
        self.df_colored = self.df.style.background_gradient(cmap="Reds", axis=1)

    def add_ue_method(self, ue_name, values):
        ue_upper_name = make_method_upper(ue_name)
        self.ue_methods.append(ue_upper_name)
        self.ue_dict[ue_upper_name] = values

    def save(self, path=None, detach_data: bool = True):
        device = self.model.device
        self._unload_model_and_tokenizer()
        if detach_data:
            data = self._detach_data()
        if path is None:
            path = "man_" + self.model.replace("/", "_") + ".dill"
        with open(path, "wb") as f:
            dill.dump(self, f)
        self._load_model_and_tokenizer(device)
        if detach_data:
            self.raw_data = data["raw_data"]
            self.data = data["data"]

    @staticmethod
    def load(path="ue_manager", update_class=True, full_restore=True, device: str = "cpu"):
        with open(path, "rb") as f:
            self = dill.load(f)
        if not update_class:
            return self

        # TODO: remove
        if "predictions" in self.ue_gen._output_dict["inference"]:
            self.ue_gen._output_dict["inference"][
                "hypotheses"
            ] = self.ue_gen._output_dict["inference"]["predictions"]
            self.ue_gen._output_dict["inference"].pop("predictions")

        args = defaultdict(lambda: None)

        if full_restore:
            args = {
                'tokenizer': self.tokenizer,
                'data': self.data,
                'raw_data': self.raw_data,
                'output_dict': self.ue_gen._output_dict
            }

        new_self = UEManager(
            ue_methods=self.ue_methods,
            model=None,
            tokenizer=args['tokenizer'],
            configs_path=self.ue_gen.configs_path,
            data=args['data'],
            raw_data=args['raw_data'],
            output_dict=args['output_dict'],
            data_config=self.data_config,
            ue_dict=self.ue_dict,
            performance_dict=self.performance_dict,
            seq2seq_metrics=self.seq2seq_metrics,
            output=self.output,
            ood_labels=getattr(self, "ood_labels", None),
            df=getattr(self, "df", None),
            tokenize=False,
            device=device,
        )
        return new_self

    @staticmethod
    def _static_load_model_and_tokenizer(
        model,
        peft_id=None,
        tokenizer=None,
        device="cpu"
    ):
        if isinstance(model, str):
            model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
        else:
            model = model.to(device)
        if peft_id is not None:
            model = PeftModel.from_pretrained(model, peft_id)
        if isinstance(tokenizer, str) or tokenizer is None:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        return model, tokenizer

    def _load_model_and_tokenizer(self, device="cpu"):
        if isinstance(self.model, str):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model).to(device)
        if isinstance(self.tokenizer, str) or self.tokenizer is None:
            if self.tokenizer_path is not None:
                tokenizer_path = self.tokenizer_path
            else:
                tokenizer_path = self.model.name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _unload_model_and_tokenizer(self):
        if not isinstance(self.model, str):
            self.model.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.name_or_path
        if not isinstance(self.tokenizer, str):
            self.tokenizer = str(self.model)

    def _get_texts_and_labels(self, inference_data, data_config=None, labels=None):
        if labels is None:
            if "input_ids" in inference_data.features:
                texts = self.raw_data[self._data_text_name]
                labels = self.raw_data[self._data_label_name]
            else:
                texts = inference_data[self._data_text_name]
                labels = inference_data[data_config["label_name"]]
        else:
            if "input_ids" in inference_data.features:
                texts = self.raw_data[self._data_text_name]
            else:
                texts = inference_data[self._data_text_name]
        return texts, labels

    @property
    def df_style(self):
        if hasattr(self, "df_colored"):
            return self.df_colored
        return self.df.style.background_gradient(cmap="Reds", axis=1)

    def _detach_data(self):
        data = {"raw_data": self.raw_data, "data": self.data}
        self.raw_data = None
        self.data = None
        return data

    def _get_method_name_and_strategy_kwargs(self, method: str, kwargs: dict):
        method_old = method
        method_name_upper = method.split("+")[0].upper()
        strategy_kwargs = {}
        generate_method = REV_OUTPUT_TYPE.get(method)
        # Check strategy kwargs
        strategy_kwargs_upd = kwargs.get(method, None) or kwargs.get(method_old, None)
        if strategy_kwargs_upd is not None:
            strategy_kwargs.update(strategy_kwargs_upd)
        if generate_method != "no":
            strategy_kwargs["inference_output"] = self.ue_gen._output_dict[
                generate_method
            ]
        if method_name_upper == "BLEUVARDET":
            strategy_kwargs["stochastic_output"] = strategy_kwargs.pop(
                "inference_output"
            )
            strategy_kwargs["deterministic_output"] = self.ue_gen._output_dict[
                "inference"
            ]
        elif method.split("-")[0] in ["MD", "NUQ", "DDU", "RDE"]:
            inference_output = strategy_kwargs.pop("inference_output")
            strategy_kwargs["train_embeddings"] = inference_output["train_embeddings"]
            strategy_kwargs["test_embeddings"] = inference_output["test_embeddings"]
        elif method == 'ENC-DIFF':
            inference_output = strategy_kwargs.pop("inference_output")
            strategy_kwargs["embedding_diffs"] = inference_output
        elif method == 'EMB-MLP':
            output = self.ue_gen._output_dict
            enc_key = "embeddings_encoder"
            dec_key = "embeddings_decoder"
            strategy_kwargs = {
                "enc_train_embeddings": output[enc_key]["train_embeddings"],
                "enc_test_embeddings": output[enc_key]["test_embeddings"],
                "dec_train_embeddings": output[dec_key]["train_embeddings"],
                "dec_test_embeddings": output[dec_key]["test_embeddings"],
                "train_preds": output[enc_key]["train_preds"]["wd_ids"],
                "test_preds": output[enc_key]["test_preds"]["wd_ids"],
                "train_labels": kwargs["embeddings"]["train_data"][self._data_label_name],
                "test_labels": kwargs["embeddings"]["test_data"][self._data_label_name]
            }
        elif any(
            (generate_method.lower().startswith(x) for x in ("ensemble", "ep_single"))
        ):
            # need to extract two keys
            if "SEQ" in method:
                strategy_kwargs["sequence_level_data"] = strategy_kwargs[
                    "inference_output"
                ]["sequence_level_data"]
            elif "PE-TOK" in method:
                strategy_kwargs["token_level_data"] = strategy_kwargs[
                    "inference_output"
                ]["pe_token_level_data"]
            elif "EP-TOK" in method:
                strategy_kwargs["token_level_data"] = strategy_kwargs[
                    "inference_output"
                ]["ep_token_level_data"]
            else:
                raise NotImplementedError
            strategy_kwargs["inference_output"] = strategy_kwargs[
                "inference_output"
            ].get("inference_output")

        return method_name_upper, strategy_kwargs
