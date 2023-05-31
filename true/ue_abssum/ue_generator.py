import gc
import logging
from copy import deepcopy, copy
from pathlib import Path
from typing import Union

import torch
from torch import cuda, Tensor
import yaml
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
)

from ue_abssum.ue_dropout import replace_dropout
from ue_abssum.default_values import REV_OUTPUT_TYPE, TOP_K
from ue_abssum.ensemble_generator import EnsembleGenerator
from ue_abssum.forward import forward
from ue_abssum.generate import generate
from ue_abssum.get_embeddings import get_embeddings
from ue_abssum.mc_utils import (
    get_mc_output,
    get_mc_forward_output,
)
from ue_abssum.utils import _load_model_and_tokenizer_if_necessary

log = logging.getLogger()


class UeGenerator:
    def __init__(
        self,
        default_method: str = "inference",
        output_dict: dict = None,
        configs_path: Union[str, Path] = None,
        seed: int = 42
    ):
        self.default_method = default_method
        self.configs_path = Path(configs_path) if configs_path is not None else None
        self._output_dict = output_dict if output_dict is not None else {}
        self.seed = seed

    def __call__(
        self, method: str = None, generate_kwargs: dict = None, **method_kwargs
    ):
        method = method or self.default_method
        return getattr(self, method)(generate_kwargs, **method_kwargs)

    def _create_output_dict(
        self, update_output_dict, generate_kwargs, ue_methods, **kwargs
    ):
        required_generate_methods = set()
        for method in ue_methods:
            required_generate_methods.add(
                REV_OUTPUT_TYPE.get(
                    method,
                )
            )

        hidden_states = []
        extract_emdeddings = False
        for generate_method in required_generate_methods:
            if update_output_dict or generate_method not in self._output_dict:
                if generate_method in ["embeddings_encoder", "embeddings_decoder"]:
                    hidden_states.append(generate_method.split("_")[-1])
                    extract_emdeddings = True

        # General inference
        if (update_output_dict or "inference" not in self._output_dict) and kwargs.pop(
            "update_inference", True
        ):
            embed_kwargs = None
            if extract_emdeddings:
                embed_kwargs = kwargs["embeddings"]
                embed_kwargs["hidden_state"] = hidden_states
            self.inference(generate_kwargs=generate_kwargs, embed_kwargs=embed_kwargs)

        # Other generations
        for generate_method in required_generate_methods:
            if update_output_dict or generate_method not in self._output_dict:
                log.info(f"Start generating output for {generate_method}...")
                if any(
                    (generate_method.startswith(x) for x in ["mc", "aug", "sampling"])
                ):
                    self.mc_aug_sampling(
                        generate_method=generate_method,
                        generate_kwargs=generate_kwargs,
                        **kwargs,
                    )
                elif generate_method == "inference_unnormalized":
                    self.inference(generate_kwargs=generate_kwargs, length_penalty=0.0)
                elif generate_method == 'embeddings_diff':
                    self.embeddings_diff(generate_kwargs=generate_kwargs, embed_kwargs=embed_kwargs)
                elif generate_method in [
                    "forward",
                    "embeddings",
                    "ensemble",
                    "ep_single",
                ]:
                    self(
                        method=generate_method,
                        generate_kwargs=generate_kwargs,
                        **kwargs,
                    )
                elif generate_method in ["embeddings_encoder", "embeddings_decoder"]:
                    log.info(
                        f"Skipped generating output for {generate_method}, generating will be in one run later."
                    )
                    continue

                log.info(f"Done with generating output for {generate_method}.")

        if (update_output_dict or extract_emdeddings) and len(hidden_states):
            log.info(
                "Start generating output for embeddings from {}...".format(
                    ",".join(hidden_states)
                )
            )
            kwargs["embeddings"]["hidden_state"] = hidden_states
            kwargs["embeddings"]["update_name"] = True
            self(
                method="embeddings",
                generate_kwargs=generate_kwargs,
                **kwargs,
            )
            log.info(
                "Done with generating output for embeddings from {}".format(
                    ",".join(hidden_states)
                )
            )

    def inference(self, generate_kwargs: dict, **kwargs):
        key = (
            "inference"
            if kwargs.get("length_penalty", 1.0) != 0
            else "inference_unnormalized"
        )

        generate_kwargs['to_eval_mode'] = True
        if generate_kwargs.get('inference_seed', None) is not None:
            dropout_rates = generate_kwargs.get('dropout_rates', [])
            inference_seed = generate_kwargs.get('inference_seed', [])
            del generate_kwargs['inference_seed'], generate_kwargs['dropout_rates']

            if len(dropout_rates) > 0:
                for p in dropout_rates:
                    print(f'Generating inference output for p={p}')
                    torch.manual_seed(int(inference_seed))
                    replace_dropout(generate_kwargs['model'], p=p)
                    p_key = f"{key}_{p}"
                    self._output_dict[p_key] = self._generate(**generate_kwargs, **kwargs)

            torch.manual_seed(int(inference_seed))
            replace_dropout(generate_kwargs['model'])

        self._output_dict[key] = self._generate(**generate_kwargs, **kwargs)

        # rewrite embeddings from "inference" to separate key in dict
        if "embeddings" in self._output_dict[key]:
            for hidden in self._output_dict[key]["embeddings"].keys():
                embeddings_name = "embeddings_{}".format(hidden)
                self._output_dict[embeddings_name] = {
                    "test_embeddings": self._output_dict[key]["embeddings"][hidden],
                }
            del self._output_dict[key]["embeddings"]

    def embeddings_diff(self, generate_kwargs: dict, **kwargs):
        output_sequences = self._output_dict["inference"]["sequences"]

        seqs = output_sequences[:, 1:]
        refined_seqs = [seq[mask].tolist() for mask, seq in zip(seqs != 0, seqs)]
        labels = generate_kwargs['data']['labels']

        data = {
            'input_ids': refined_seqs,
            'labels': labels
        }
        _args = copy(generate_kwargs)
        _args['data'] = Dataset.from_dict(data)

        output_embs = self._generate(**_args, **kwargs)["embeddings"]["encoder"]
        input_embs = self._output_dict["embeddings_encoder"]["test_embeddings"]
        diff = output_embs - input_embs
        self._output_dict["embeddings_diff"] = diff

    def forward(self, generate_kwargs: dict, **kwargs):
        if "inference" not in self._output_dict:
            self.inference(generate_kwargs)

        # Change labels in data to the generated ones
        orig_data = generate_kwargs["data"]
        data_pl = self._prepare_data_for_forward(
            orig_data,
            self._output_dict["inference"]["sequences"],
            generate_kwargs["tokenizer"],
        )
        generate_kwargs["data"] = data_pl
        # Forward pass with pseudo-labeled data
        self._output_dict["forward"] = self._mc_forward(generate_kwargs, kwargs)
        # Return the original data
        generate_kwargs["data"] = orig_data

    def mc_aug_sampling(self, generate_method: str, generate_kwargs: dict, **kwargs):
        self._output_dict[generate_method] = self._mc_generate(
            generate_kwargs, generate_method, kwargs
        )

    def embeddings(self, generate_kwargs: dict, **kwargs):
        emb_kwargs = kwargs.pop("embeddings")
        if (emb_kwargs is not None) and ("model_name" in emb_kwargs.keys()):
            # Get embeddings from another model
            get_embs_kwargs = {
                "model": AutoModel.from_pretrained(emb_kwargs["model_name"]).to(
                    generate_kwargs["model"].device
                ),
                "tokenizer": AutoTokenizer.from_pretrained(emb_kwargs["model_name"]),
                "data_is_tokenized": False,
                "prepare_model": False,
                "use_automodel": True,
                "to_numpy": False,
                "document_name": generate_kwargs["data_config"]["text_name"],
                "extraction_method": "forward",
            }
        else:
            # Get embeddings from the main model
            hidden_state = emb_kwargs.get("hidden_state", "encoder")
            get_embs_kwargs = {
                "model": generate_kwargs["model"],
                "tokenizer": generate_kwargs["tokenizer"],
                "data_is_tokenized": False,
                "prepare_model": False,
                "use_automodel": True,
                "to_numpy": False,
                "batch_size": generate_kwargs.get("batch_size", 16),
                "hidden_state": hidden_state,
                "use_averaging": emb_kwargs.get("use_averaging", True),
                "ignore_padding": emb_kwargs.get("ignore_padding", True),
                "generation_max_length": generate_kwargs.get(
                    "generation_max_length", 100
                )
                if "decoder" in hidden_state
                else 3,
                "document_name": generate_kwargs["data_config"]["text_name"],
                "extraction_method": "generate",
                "all_layers": emb_kwargs.get("all_layers", False),
                "aggregation_method": emb_kwargs.get("aggregation_method", "mean"),
            }

        if emb_kwargs["update_name"]:
            hidden = hidden_state if isinstance(hidden_state, str) else hidden_state[-1]
            embeddings_name = "embeddings_{}".format(hidden)
        else:
            embeddings_name = "embeddings"

        train_embeddings = get_embeddings(
            dataloader_or_data=emb_kwargs["train_data"], **get_embs_kwargs
        )

        test_embeddings = None
        if embeddings_name not in self._output_dict.keys():
            test_embeddings = get_embeddings(
                dataloader_or_data=emb_kwargs["test_data"],
                **get_embs_kwargs,
            )

        _args = {
            "tokenizer": generate_kwargs["tokenizer"],
            "is_tokenized": False
        }

        train_generate_kwargs = copy(generate_kwargs)
        test_generate_kwargs = copy(generate_kwargs)

        train_generate_kwargs["data"] = emb_kwargs["train_data"]
        test_generate_kwargs["data"] = emb_kwargs["test_data"]

        train_generate_kwargs.update(_args)
        test_generate_kwargs.update(_args)

        train_preds = self._generate(**train_generate_kwargs)
        test_preds = self._generate(**test_generate_kwargs)

        for hidden in train_embeddings.keys():
            if emb_kwargs["update_name"]:
                embeddings_name = "embeddings_{}".format(hidden)
            else:
                embeddings_name = "embeddings"
            if embeddings_name not in self._output_dict.keys():
                self._output_dict[embeddings_name] = {
                    "train_embeddings": train_embeddings[hidden],
                    "test_embeddings": test_embeddings[hidden],
                    "train_preds": train_preds,
                    "test_preds": test_preds
                }
            else:
                self._output_dict[embeddings_name][
                    "train_embeddings"
                ] = train_embeddings[hidden]
                self._output_dict[embeddings_name][
                    "train_preds"
                ] = train_preds
                self._output_dict[embeddings_name][
                    "test_preds"
                ] = test_preds

        del get_embs_kwargs
        cuda.empty_cache()
        gc.collect()

    def ensemble(self, generate_kwargs: dict, **kwargs):
        kwargs = kwargs.get("ensemble")
        assert kwargs is not None, "Please provide the de_model_paths."
        device = generate_kwargs["model"].device
        generate_kwargs["model"].to("cpu")
        model = (
            EnsembleGenerator.from_pretrained(generate_kwargs["model"].name_or_path)
            .train()
            .to(device)
        )

        de_models_paths = kwargs.get("de_models_paths")
        if de_models_paths is None:
            log.info("Argument `de_models_paths` is not provided. Using MC ensemble.")
            dropout_rate = kwargs.get("ensemble_dropout_rate", 0.1)
            dropout_module = kwargs.get("dropout_module", "whole")
            if dropout_module == "decoder":
                replace_dropout(model.decoder, p=dropout_rate, share_across_tokens=True)
            elif dropout_module == "encoder":
                replace_dropout(model.encoder, p=dropout_rate, share_across_tokens=True)
            else:
                replace_dropout(model, p=dropout_rate, share_across_tokens=True)
            model.mc = True
            model.mc_models_num = kwargs.get("mc_iterations", 5)
            model.base_seed = self.seed
            # Disable train mode since custom dropout doesn't need it
            generate_kwargs["to_eval_mode"] = True
        else:
            model.add_ensemble_models(*de_models_paths)

        model.ensembling_mode = kwargs.get("ensembling_mode", "pe")
        print("Ensembling mode is ", model.ensembling_mode)

        # We make an assumption that all the models have the same tokenizer
        ensemble_generate_kwargs = copy(generate_kwargs)
        for key in ensemble_generate_kwargs:
            # if upd_value := kwargs.get(key) is not None:
            if key in kwargs:
                ensemble_generate_kwargs[key] = kwargs[key]
        ensemble_generate_kwargs["aggregate_sequences_scores"] = False
        ensemble_generate_kwargs["model"] = model
        output_dict = self._generate(**ensemble_generate_kwargs, using_ensemble=True)

        del ensemble_generate_kwargs["model"], model
        cuda.empty_cache()
        gc.collect()

        #num_obs = len(generate_kwargs["data"])
        #num_beams_per_obs = generate_kwargs["num_return_sequences"]
        #max_seq_len = output_dict["sequences"].shape[-1]
        ## Teacher forcing: sequence level data and EP token level data
        #models = []
        #if de_models_paths is None:
        #    model_path = generate_kwargs["model"].name_or_path
        #    models.append(
        #        AutoModelForSeq2SeqLM.from_pretrained(model_path).train().to(device)
        #    )
        #    replace_dropout(models[0], share_across_tokens=True)
        #    ensemble_generate_kwargs["to_eval_mode"] = True
        #else:
        #    models_paths = [generate_kwargs["model"].name_or_path] + list(
        #        de_models_paths
        #    )
        #    for model_path in tqdm(models_paths, desc="DE models uploaded..."):
        #        models.append(
        #            AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        #        )
        #ensemble_generate_kwargs.update(
        #    {
        #        "model": models,
        #        "tokenizer": generate_kwargs["tokenizer"],
        #        "data": self._prepare_data_for_forward(
        #            generate_kwargs["data"],
        #            output_dict["sequences"].reshape(-1, max_seq_len),
        #            ensemble_generate_kwargs["tokenizer"],
        #        ),
        #        "max_seq_len": output_dict["max_seq_length"],
        #        "to_numpy": True,
        #        "to_eval_mode": False,
        #        "mc": True if de_models_paths is None else False,
        #        "mc_iterations": kwargs.get("mc_iterations", 5),
        #        "seed": self.seed
        #    }
        #)

        #models_output_dict = self._forward(**ensemble_generate_kwargs)

        sequence_level_data = {
            "probas": output_dict["probas"].transpose(1, 0, 2),
            "log_probas": output_dict["log_probas"].transpose(1, 0, 2)
        }

        #del ensemble_generate_kwargs, models
        #cuda.empty_cache()
        #gc.collect()

        generate_kwargs["model"].to(device)

        self._output_dict["ensemble"] = {
            "hypotheses": [x[0] for x in output_dict["hypotheses"]],
            "inference_output": output_dict,
            "pe_token_level_data": output_dict["pe_token_level_scores"],
            "ep_token_level_data": output_dict["ep_token_level_scores"],
            "sequence_level_data": sequence_level_data
        }

    def ep_single(self, generate_kwargs: dict, **kwargs):
        kwargs = kwargs.get("ep_single") or kwargs.get("ensemble") or {}
        assert kwargs is not None, "Please provide the de_model_paths."
        device = generate_kwargs["model"].device
        model = (
            EnsembleGenerator.from_pretrained(generate_kwargs["model"].name_or_path)
            .train()
            .to(device)
        )
        # We make an assumption that all the models have the same tokenizer
        ensemble_generate_kwargs = deepcopy(generate_kwargs)
        for key in ensemble_generate_kwargs:
            # if upd_value := kwargs.get(key) is not None:
            if key in kwargs:
                ensemble_generate_kwargs[key] = kwargs[key]
        ensemble_generate_kwargs["aggregate_sequences_scores"] = False
        ensemble_generate_kwargs["model"] = model

        num_obs = len(generate_kwargs["data"])
        num_beams_per_obs = generate_kwargs["num_return_sequences"]
        output_dict = self._output_dict["inference"]
        max_seq_len = output_dict["sequences_wo_agg"].shape[-1]
        # Teacher forcing: sequence level data and EP token level data
        de_models_paths = kwargs.get("de_models_paths")
        models = []
        if de_models_paths is None:
            log.info("Argument `de_models_paths` is not provided. Using MC ensemble.")
            model_path = generate_kwargs["model"].name_or_path
            for _ in tqdm(
                range(kwargs.get("mc_iterations", 5)),
                desc="DE models forward uploaded...",
            ):
                models.append(
                    AutoModelForSeq2SeqLM.from_pretrained(model_path).train().to(device)
                )
        else:
            models_paths = [generate_kwargs["model"].name_or_path] + list(
                de_models_paths
            )
            for model_path in tqdm(models_paths, desc="DE models uploaded..."):
                models.append(
                    AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                )
        ensemble_generate_kwargs.update(
            {
                "model": models,
                "tokenizer": generate_kwargs["tokenizer"],
                "data": self._prepare_data_for_forward(
                    generate_kwargs["data"],
                    output_dict["sequences_wo_agg"].reshape(-1, max_seq_len),
                    ensemble_generate_kwargs["tokenizer"],
                ),
                "max_seq_len": output_dict["max_seq_length"],
                "to_numpy": True,
                "to_eval_mode": False,
            }
        )
        models_output_dict = self._forward(**ensemble_generate_kwargs)
        sequence_level_data = {
            "probas": models_output_dict["probas"].reshape(
                len(models), num_obs, num_beams_per_obs
            ),
            "log_probas": models_output_dict["log_probas"].reshape(
                len(models), num_obs, num_beams_per_obs
            ),
            "log_probas_iter": models_output_dict["log_probas_iter"].reshape(
                len(models), num_obs, num_beams_per_obs, -1
            ),
            "num_labels": models_output_dict["num_labels"].reshape(
                num_obs, num_beams_per_obs
            ),
        }
        ep_token_level_data = {
            k: v.reshape(num_obs, num_beams_per_obs)
            for k, v in models_output_dict["token_level_measures"].items()
        }
        # Reshape all measures wrt num_sequences
        del ensemble_generate_kwargs, models
        cuda.empty_cache()
        gc.collect()

        # TODO: exclude useless duplication of inference_output
        self._output_dict["ep_single"] = {
            "inference_output": output_dict,
            "ep_token_level_data": ep_token_level_data,
            "sequence_level_data": sequence_level_data,
            "models_output": models_output_dict,
        }

    @staticmethod
    def _generate(**generate_kwargs):
        log.info("Generating general inference...")
        generate_kwargs = _load_model_and_tokenizer_if_necessary(generate_kwargs)
        output = generate(**generate_kwargs)
        log.info("Done.")
        return output

    @staticmethod
    def _forward(**generate_kwargs):
        log.info("Forwarding general inference...")
        generate_kwargs = _load_model_and_tokenizer_if_necessary(generate_kwargs)
        output = forward(**generate_kwargs)
        log.info("Done.")
        return output

    @staticmethod
    def _update_generate_kwargs(generate_kwargs, generate_method, kwargs, configs_path):
        if configs_path is None:
            return {}
        mc_or_sampling = generate_method.split("_")[0]
        with open(configs_path / (mc_or_sampling + ".yaml")) as f:
            config = yaml.load(f, yaml.Loader)
        ensemble_or_beam = generate_method.split("_")[1]
        with open(configs_path / (ensemble_or_beam + ".yaml")) as f:
            config.update(yaml.load(f, yaml.Loader))
        # Update
        keys_to_pop = []
        for key in config:
            if key in generate_kwargs:
                keys_to_pop.append(key)
            elif key in kwargs:
                config[key] = kwargs[key]
        # Remove keys to pop
        for key in keys_to_pop:
            config.pop(key)
        return config

    def _mc_generate(self, generate_kwargs, generate_method, kwargs):
        generate_kwargs = _load_model_and_tokenizer_if_necessary(generate_kwargs)
        config = self._update_generate_kwargs(
            generate_kwargs, generate_method, kwargs, self.configs_path
        )
        if "beam" in generate_method:
            config["mc_iterations"] = 1
        log.info(f"Generating {generate_method} inference...")
        mc_output = get_mc_output(**generate_kwargs, **config)
        log.info("Done.")
        return mc_output

    def _mc_forward(self, generate_kwargs, kwargs):
        generate_kwargs = _load_model_and_tokenizer_if_necessary(generate_kwargs)
        mc_iterations = kwargs.get("mc_iterations")
        if mc_iterations is None and self.configs_path is not None:
            with open(self.configs_path / "mc.yaml") as f:
                config = yaml.load(f, yaml.Loader)
            mc_iterations = config["mc_iterations"]
        log.info(f"Making forward pass...")
        mc_output = get_mc_forward_output(
            **generate_kwargs, mc_iterations=mc_iterations
        )
        log.info("Done.")
        return mc_output

    @staticmethod
    def _prepare_data_for_forward(data, sequences, tokenizer):
        # Substitute padding idx with -100
        old_pad_idx = tokenizer.pad_token_id
        new_pad_idx = DataCollatorForSeq2Seq.label_pad_token_id
        # From the first element since the 0th is a token to start the generation
        generated_sequences = sequences[:, 1:]
        generated_sequences[generated_sequences == old_pad_idx] = new_pad_idx
        # Convert to numpy the sequences since it must be stored in such a format in the `Dataset`
        if isinstance(generated_sequences, Tensor):
            generated_sequences = generated_sequences.cpu().numpy()
        if "labels" in data.features:
            data_pl = data.remove_columns("labels")
        else:
            data_pl = deepcopy(data)
        # Length of `generated_sequences` aligns with the data length
        if len(data) == len(generated_sequences):
            data_pl = data_pl.add_column("labels", generated_sequences.tolist())
        # When `generated_sequences` contains several samples for each instance
        else:
            num_samples_per_inst = len(generated_sequences) // len(data)
            input_ids = [
                x for x in data_pl["input_ids"] for _ in range(num_samples_per_inst)
            ]
            attention_mask = [
                x
                for x in data_pl["attention_mask"]
                for _ in range(num_samples_per_inst)
            ]
            data_pl = Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": generated_sequences.tolist(),
                }
            )
        return data_pl
