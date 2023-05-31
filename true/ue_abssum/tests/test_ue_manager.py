from ue_abssum.ue_manager import UEManager

import os

from datasets import Dataset
import numpy as np
from pytest import fixture
import transformers


DATASET = Dataset.from_dict(
    {
        "document": ["aaa", "bbb", "ccc", "ddd"],
        "summary": ["a", "b", "c", "d"],
    }
)


@fixture
def ue_manager():
    transformers.set_seed(42)
    os.environ["DEVICE_FOR_DATA_RESTORING"] = "cpu"
    yield UEManager(["NSP"], "hf-internal-testing/tiny-random-bart", data=DATASET)


def test_add_ue_method(ue_manager):
    ue_manager.add_ue_method("ue_method", [1, 2, 3, 4])
    assert ue_manager.ue_methods == ["NSP", "UE_METHOD"]
    assert ue_manager.ue_dict == {"UE_METHOD": [1, 2, 3, 4]}


def test_get_ue_dict(ue_manager):
    # Launch with default ue_methods
    result = ue_manager.get_ue_dict()
    assert set(result.keys()) == {"NSP"}
    assert len(result["NSP"]) == 4

    # Launch with ue_methods as string
    result = ue_manager.get_ue_dict(ue_methods="MSP")
    assert set(result.keys()) == {"NSP", "MSP"}
    assert len(result["MSP"] == 4)

    # Launch with update_ue_dict = True
    result = ue_manager.get_ue_dict(ue_methods="MSP", update_ue_dict=True)
    assert set(result.keys()) == {"MSP"}
    assert len(result["MSP"] == 4)

    # Launch with ue_methods as array
    result = ue_manager.get_ue_dict(ue_methods=["NSP", "MSP", "USP"], update_ue_dict=True)
    assert set(result.keys()) == {"NSP", "MSP", "USP"}
    assert len(result["MSP"] == 4)


def test_calculate_seq2seq_metrics(ue_manager):
    ue_manager.get_ue_dict()

    # Launch with one metric
    metrics, result = ue_manager.calculate_seq2seq_metrics(metrics_to_use=["BLEU"])
    assert set(metrics["inference"].keys()) == {"BLEU"}
    assert len(metrics["inference"]["BLEU"]) == 4
    assert set(result["inference"].keys()) == {"BLEU"}
    assert isinstance(result["inference"]["BLEU"], float)

    # Launch with multiple metrics
    metrics, result = ue_manager.calculate_seq2seq_metrics(metrics_to_use=["BLEU", "ROUGE"])
    assert set(metrics["inference"].keys()) == {"BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"}
    assert set(result["inference"].keys()) == {"BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"}


def test_calculate_scores(ue_manager):
    ue_manager.get_ue_dict()
    ue_manager.calculate_seq2seq_metrics(metrics_to_use=["BLEU", "ROUGE"])

    result, _ = ue_manager.calculate_scores(seq2seq_metrics_names=["BLEU", "ROUGE-1"], plot=False)
    for seq2seq_metric in ["BLEU", "ROUGE-1"]:
        for metric, scores_len in zip(["prr", "rcc"], [5, 4]):
            assert f"{seq2seq_metric}_NSP_{metric}_target_score" in result
            for ue in ["NSP", "rand", "oracle"]:
                assert f"{seq2seq_metric}_{ue}_{metric}_score" in result
                assert f"{seq2seq_metric}_{ue}_{metric}_scores" in result
                assert len(result[f"{seq2seq_metric}_{ue}_{metric}_scores"]) == scores_len


def numpy_to_list(dictionary):
    return {k: list(v) if isinstance(v, np.ndarray) else v for k, v in dictionary.items()}


def test_call(ue_manager):
    result, _ = ue_manager(metrics_to_use=["BLEU"], plot=False)

    ue_manager2 = UEManager(["NSP"], "hf-internal-testing/tiny-random-bart", data=DATASET)
    ue_dict = ue_manager2.get_ue_dict()
    seq2seq_metrics, _ = ue_manager2.calculate_seq2seq_metrics(metrics_to_use=["BLEU"])
    result2, _ = ue_manager2.calculate_scores(plot=False)

    assert numpy_to_list(ue_dict) == numpy_to_list(ue_manager.ue_dict)
    assert numpy_to_list(seq2seq_metrics["inference"]) == numpy_to_list(ue_manager.seq2seq_metrics["inference"])
    assert numpy_to_list(result) == numpy_to_list(result2)


def test_save_load(ue_manager):
    ue_manager(metrics_to_use=["BLEU"], plot=False)
    ue_manager.save(path="manager")
    ue_manager2 = UEManager.load("manager")
    os.remove("manager")

    assert numpy_to_list(ue_manager.ue_dict) == numpy_to_list(ue_manager2.ue_dict)
    assert numpy_to_list(ue_manager.seq2seq_metrics["inference"]) == numpy_to_list(ue_manager2.seq2seq_metrics["inference"])
