"""
This module is a main entrypoint for running UE experiments with T5 models
"""

import os
import pickle
import logging
from pathlib import Path

import hydra
from hydra.initialize import GlobalHydra
from omegaconf import OmegaConf

import numpy as np
from numpy.random import default_rng
from transformers import AutoTokenizer
from tqdm import tqdm

from true.utils import log_config
from true.main_decorator import main_decorator
from true.prefix_trie import MarisaTrie


log = logging.getLogger()

OmegaConf.register_new_resolver(
    "to_string",
    lambda x: x.replace("/", "_").replace("-", "_"),
    replace=True,
)

OmegaConf.register_new_resolver(
    "get_home_path",
    lambda x: Path("/".join(os.path.abspath(__file__).split("/")[:2])),
    replace=True,
)

MAX_TRAIN_DATA_SIZE = int(os.environ.get("MAX_TRAIN_DATA_SIZE", 20_000))


def build_trie(titles, tokenizer):
    """
    Builds prefix trie from vocab
    """
    allowed_names_en = [name for lang, name in titles.keys() if lang == 'en']
    allowed_names_en_tok = [tokenizer(name)['input_ids'] for name in tqdm(allowed_names_en)]
    tok_names_padded = [[tokenizer.pad_token_id,] + toks for toks in allowed_names_en_tok if tokenizer.unk_token_id not in toks]

    print(len(tok_names_padded))
    
    trie = MarisaTrie(sequences=tok_names_padded)
    
    return trie


@main_decorator
def run_ue(config, work_dir):
    """
    Main part of the script
    """
    # Imports inside function to set environment variables before imports
    from true.data.load_data import load_data
    from true.add_ood import add_ood_from_dataset, corrupt_row
    from true.ue_manager import UEManager
    from true.utils import get_generation_max_length
    from true.default_values import DEFAULT_UE_METHODS

    # Log config so that it is visible from the console
    log_config(log, config)
    log.info("Loading data...")
    cache_dir = config.cache_dir
    train_data, dev_data, test_data = load_data(
        config.data,
        cache_dir,
    )

    if config.get('corrupt_data', None) is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.path, truncation=True, padding=True
        )
        gen = default_rng(seed=config.seed)
        test_data = test_data.map(
            lambda row: corrupt_row(row, config.data.text_name, tokenizer, gen),
            load_from_cache_file=False,
        )

    ood_detection = hasattr(config, "ood")
    if ood_detection:
        log.info("Loading OOD dataset...")
        _, _, test_ood_data = load_data(config.ood_data, cache_dir)
        log.info("Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.ood.tokenizer, truncation=True, padding=True
        )
        log.info("Preparing OOD dataset...")
        gen = default_rng(seed=config.seed)
        test_data = add_ood_from_dataset(
            test_data,
            test_ood_data,
            gen,
            corrupt_samples=config.ood.corrupt,
            text_field=config.data.text_name,
            ood_text_field=config.ood_data.text_name,
            label_field=config.data.label_name,
            ood_label_field=config.ood_data.label_name,
            shuffle=config.ood.shuffle,
            tokenizer=tokenizer,
        )

        with open(work_dir / 'ood_labels.pkl', 'wb') as handle:
            pickle.dump(test_data['label'], handle)

    vocab_filename = "lang_title2wikidataID-normalized_with_redirect_en.pkl"
    if hasattr(config, "resolve_sequences") and config.resolve_sequences == True:
        log.info("Loading KG text to id mapping...")
        with open(Path(config.output_dir) / vocab_filename, "rb") as handle:
            kg_id_mapping = pickle.load(handle)
    else:
        kg_id_mapping = None

    if hasattr(config, "prefix_tree") and config.prefix_tree == True:
        filename = "titles_lang_all105_marisa_trie_with_redirect_for_t5.pkl"
        filepath = Path(config.output_dir) / filename
        if os.path.isfile(filepath):
            log.info("Loading prefix trie...")
            with open(Path(config.output_dir) / filename, "rb") as handle:
                trie = pickle.load(handle)
        elif os.path.isfile(Path(config.output_dir) / vocab_filename):
            log.info("Building prefix trie...")
            if kg_id_mapping is None:
                log.info("Loading vocab...")
                with open(Path(config.output_dir) / vocab_filename, "rb") as handle:
                    kg_id_mapping = pickle.load(handle)

            tokenizer = AutoTokenizer.from_pretrained(config.model.name)
            trie = build_trie(kg_id_mapping, tokenizer)
            with open(Path(config.output_dir) / filename, "wb") as handle:
                pickle.dump(trie, handle)
        else:
            raise ValueError("prefix_tree is set to true, but there is no existing trie file present and no vocabulary file present to construct trie from")

        def prefix_allowed_tokens_fn(batch_id, sent):
            return trie.get(sent.tolist()),

    else:
        prefix_allowed_tokens_fn = None

    try:
        ue_type = config.ue_methods.type
        if ue_type == 'single':
            ue_methods = config.ue_methods.single_names
        elif ue_type == 'ensemble':
            ue_methods = config.ue_methods.ensemble_names
        elif ue_type == 'cuda_ensemble':
            ue_methods = config.ue_methods.cuda_ensemble_names
        elif ue_type == 'mc_ensemble':
            ue_methods = config.ue_methods.mc_ensemble_names
        elif ue_type == 'embeddings':
            ue_methods = config.ue_methods.embeddings_names
        else:
            ue_methods = DEFAULT_UE_METHODS


        print("ue_methods:", ue_methods)

        GlobalHydra.instance().clear()
        hydra.initialize(
            config_path=str(
                Path(os.environ.get("HYDRA_CONFIG_PATH", "configs")) / "model_seed"
            )
        )


        print("path:", str(Path(os.environ.get("HYDRA_CONFIG_PATH", "configs")) / "model_seed"))

        for seed in config.model_seeds:
            model_path = config.model.path
            tokenizer_path = config.get("tokenizer_path", None)
            if tokenizer_path is None:
                tokenizer_path = model_path

            log.info(
                f"Constructing the UE Manager with seed: {seed}, UE methods: {ue_methods}..."
            )

            man = UEManager(
                model=model_path,
                tokenizer=tokenizer_path,
                data=test_data.select(range(100)),
                seed=seed,
                ue_methods=ue_methods,
                data_config=config.data,
                configs_path=config.generation_configs_path,
                device=str(config.model.device)
            )
            log.info("Done with constructing the UE Manager.")
            generation_max_length = (
                config.model.generation_max_length
                or get_generation_max_length(train_data, man.tokenizer, config.data)
            )
            log.info("Generating predictions for the test set...")
            # emb_model_path = f"Aktsvigun/bert-base-{config.data.dataset_name}"
            # Set methods kwargs
            ue_methods_kwargs = getattr(config.ue_methods, "kwargs", {}) or {}

            inference_seed = getattr(config, 'inference_seed', None)
            num_beam_groups = getattr(config, 'num_beam_groups', None)
            diversity_penalty = getattr(config, 'diversity_penalty', None)
            repetition_penalty = getattr(config, 'repetition_penalty', None)
            do_sample = getattr(config, 'do_sample', None)
            top_k = getattr(config, 'top_k', None)
            top_p = getattr(config, 'top_p', None)
            dropout_module = getattr(config.model, 'dropout_module', None)
            dropout_rates = getattr(config, 'dropout_rates', [])
            if hasattr(config.ue_methods, 'ensemble'):
                ensembling_mode = getattr(config.ue_methods.ensemble, 'ensembling_mode', None)
                ensemble_dropout_rate = getattr(config.ue_methods.ensemble, 'dropout_rate', 0.1)
            else:
                ensembling_mode = None
                ensemble_dropout_rate = None

            if ue_type == 'embeddings':
                emb_args = {
                    "train_data": train_data,
                    "test_data": test_data,
                }
            else:
                emb_args = None

            de_model_paths = None
            # de_model_paths = config.ue_methods.ensemble.de_model_paths
            # if de_model_paths is None:
            #     de_model_paths = [
            #         f"{'_'.join(model_path.split('_')[:-1])}_{de_seed}"
            #         for de_seed in seed_config["de_model_seeds"]
            #     ]
            # Necessary for embeddings (mainly RDE)
            if len(train_data) > MAX_TRAIN_DATA_SIZE:
                train_data = train_data.train_test_split(
                    train_size=MAX_TRAIN_DATA_SIZE,
                    shuffle=True,
                    seed=seed,
                    load_from_cache_file=False,
                )["train"]
            man(
                generation_max_length=generation_max_length,
                seed=seed,
                batch_size=config.model.batch_size,
                num_return_sequences=config.model.num_return_sequences,
                embeddings=emb_args,
                ensemble={
                    "de_models_paths": de_model_paths,
                    "batch_size": config.ue_methods.ensemble.batch_size,
                    "ensembling_mode": ensembling_mode,
                    "ensemble_dropout_rate": ensemble_dropout_rate,
                    "dropout_module": dropout_module
                },
                metrics_to_use=config.get("metrics_to_use"),
                ood_labels=np.array(test_data["label"]) if ood_detection else None,
                kg_id_mapping=kg_id_mapping,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                inference_seed=inference_seed,
                dropout_rates=dropout_rates,
                diversity_penalty=diversity_penalty,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                num_beam_groups=num_beam_groups,
                **ue_methods_kwargs,
            )
            log.info("Saving...")
            # Save the manager
            man.save(work_dir / f"{man.model.name_or_path.replace('/', '_')}_{seed}")

    except Exception as e:
        log.error(e, exc_info=True)
    finally:
        log.info("Finished.")


@hydra.main(
    config_path=os.environ.get("HYDRA_CONFIG_PATH", "configs"),
    config_name=os.environ.get("HYDRA_CONFIG_NAME"),
)
def main(config):
    assert (
        os.environ.get("HYDRA_CONFIG_NAME") is not None
    ), "Hydra config name is not specified"
    run_ue(config)


if __name__ == "__main__":
    main()
