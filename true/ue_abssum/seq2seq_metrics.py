import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Union

import numpy as np
from datasets import load_metric
from nltk.stem import porter
from nltk.tokenize import sent_tokenize
from rouge_score import tokenize
from tqdm import tqdm

from ue_abssum.seq2seq_metrics_utils import pair_bleu, get_bart_scores

log = logging.getLogger()

if os.environ.get("USE_ADD_SEQ2SEQ_METRICS", True):
    try:
        path = Path("/".join(os.path.abspath(__file__).split("/")[:-1]))
        sys.path.append(str(path / "packages/summac"))
        sys.path.append(str(path / "packages/BARTScore"))
        from .packages.bart_score import BARTScorer
        from .packages.summac.summac.model_summac import SummaCZS, SummaCConv

        ADD_METRICS_IMPORTED = True
    except Exception as e:
        log.info(f"Cannot load consistency metrics: {e}")
        ADD_METRICS_IMPORTED = False


def calculate_rouge(
    predictions: List[str],
    labels: List[str],
    is_zeroword: np.ndarray = None,
    is_uniword: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate ROUGE scores
    :param predictions:
    :param labels:
    :param is_zeroword:
    :param is_uniword:
    :return: dictionary with 3 values each of len `num_obs`
    """
    if is_zeroword is None:
        is_zeroword = np.zeros(len(predictions), dtype=bool)
    if is_uniword is None:
        is_uniword = np.zeros(len(predictions), dtype=bool)

    rouge = load_metric("rouge")
    rouges = rouge.compute(
        predictions=predictions,
        references=labels,
        use_stemmer=True,
        use_aggregator=False,
    )
    # Do not need ROUGE-L-sum
    metrics = np.array([[x.fmeasure for x in value] for value in rouges.values()])[:3]
    # Substitute invalid observations with nans
    metrics[0][is_zeroword] = metrics[2][is_zeroword] = np.nan
    metrics[1][is_zeroword | is_uniword] = np.nan
    metrics = {
        "ROUGE-1": metrics[0],
        "ROUGE-2": metrics[1],
        "ROUGE-L": metrics[2],
    }
    return metrics


def calculate_bleu(predictions: List[str], labels: List[str]) -> Dict[str, np.ndarray]:
    return {
        "BLEU": np.array(
            [pair_bleu(pred, label) for pred, label in tqdm(zip(predictions, labels))]
        )
    }


def calculate_summac(
    predictions: List[str], labels: List[str], texts: List[str]
) -> Dict[str, np.ndarray]:
    # decoded_texts = ["\n".join(sent_tokenize(text.strip())) for text in texts]
    start_time = time.time()
    device = os.environ.get("DEVICE_FOR_DATA_RESTORING", "cuda")
    cons_model = SummaCConv(granularity="sentence", device=device)
    preds_score = np.array(cons_model.score(texts, predictions)["scores"])
    labels_score = np.array(cons_model.score(texts, labels)["scores"])
    rel_score = preds_score / labels_score
    log.info(f"SummaC took {time.time() - start_time:.4f} seconds")
    return {"SUMMAC-pred": preds_score, "SUMMAC-rel": rel_score}


def calculate_bartscore(
    predictions: List[str],
    labels: List[str],
    texts: List[str],
    directions: Union[Tuple[str], List[str]] = ("hr", "fa"),
    batch_size=4,
    **bartscorer_init_params,
) -> Dict[str, np.ndarray]:
    # decoded_texts = ["\n".join(sent_tokenize(text.strip())) for text in texts]
    start_time = time.time()

    scorer = BARTScorer(**bartscorer_init_params)
    scores = get_bart_scores(
        scorer,
        preds=predictions,
        refs=labels,
        texts=texts,
        directions=directions,
        batch_size=batch_size,
    )
    log.info(f"BARTScore took {time.time() - start_time:.4f} seconds")
    return scores


def prepare_predictions_and_labels_for_rouge(
    predictions: List[str], labels: List[str]
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    predictions = ["\n".join(sent_tokenize(pred.strip())) for pred in predictions]
    labels = ["\n".join(sent_tokenize(label.strip())) for label in labels]
    # Calculate zero- and uni-word hypotheses (that must be excluded from calculating
    # ROUGE-1 and ROUGE-L for the former and ROUGE-2 for both)
    stemmer = porter.PorterStemmer()
    tokenized_labels = [tokenize.tokenize(x, stemmer) for x in labels]
    len_tokenized_labels = np.array([len(x) for x in tokenized_labels])
    is_uniword = len_tokenized_labels == 1
    is_zeroword = len_tokenized_labels == 0
    return predictions, labels, is_zeroword, is_uniword


def calculate_top_1_acc(
    predictions, labels
) -> Dict[str, np.ndarray]:
    start_time = time.time()
    assert(len(predictions) == len(labels))
    corr = []
    for pred, label in tqdm(zip(predictions, labels)):
        if isinstance(label, str):
            corr.append(1 if pred[0] == label else 0)
        elif isinstance(label, list):
            corr.append(1 if pred[0] in label else 0)
    log.info(f"Top 1 acc took {time.time() - start_time:.4f} seconds")
    return {"Top 1 Acc": corr}


def calculate_top_all_acc(
    predictions, labels
) -> Dict[str, np.ndarray]:
    start_time = time.time()
    assert(len(predictions) == len(labels))
    corr = []
    for pred, label in tqdm(zip(predictions, labels)):
        if isinstance(label, str):
            corr.append(1 if label in pred else 0)
        elif isinstance(label, list):
            inter = set(label).intersection(set(pred))
            corr.append(1 if len(inter) > 0 else 0)
    log.info(f"Top all acc took {time.time() - start_time:.4f} seconds")
    return {"Top all Acc": corr}


def calculate_mrr(
    predictions, labels
) -> Dict[str, np.ndarray]:
    start_time = time.time()
    assert(len(predictions) == len(labels))
    rrs = []
    for pred, label in tqdm(zip(predictions, labels)):
        if isinstance(label, str):
            if label in pred:
                rank = np.where(np.array(pred) == label)[0][0].item() + 1
                rrs.append(1 / rank)
            else:
                rrs.append(0)
        elif isinstance(label, list):
            inter = set(label).intersection(set(pred))
            if len(inter) == 0:
                rrs.append(0)
            else:
                top_rank = max([(np.where(np.array(pred) == id_)[0][0].item() + 1) for id_ in inter])
                rrs.append(1 / top_rank)
    log.info(f"MRR took {time.time() - start_time:.4f} seconds")
    return {"MRR": rrs}

