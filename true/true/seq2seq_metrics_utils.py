import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import corpus_bleu


def smoothing_function(p_n, references, hypothesis, hyp_len):
    """
    Smooth-BLEU (BLEUS) as proposed in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
    smoothed_p_n = []
    for i, p_i in enumerate(p_n, start=1):
        # Smoothing is not applied for unigrams
        if i > 1:
            # If hypothesis length is lower than the current order, its value equals (0 + 1) / (0 + 1) = 0
            if hyp_len < i:
                assert p_i.denominator == 1
                smoothed_p_n.append(1)
            # Otherwise apply smoothing
            else:
                smoothed_p_i = (p_i.numerator + 1) / (p_i.denominator + 1)
                smoothed_p_n.append(smoothed_p_i)
        else:
            smoothed_p_n.append(p_i)
    return smoothed_p_n


def pair_bleu(references, prediction):
    """
    Compute the bleu score between two given texts.
    A smoothing function is used to avoid zero scores when
    there are no common higher order n-grams between the
    texts.
    """
    tok_ref = [word_tokenize(s) for s in sent_tokenize(references)]
    tok_pred = [word_tokenize(s) for s in sent_tokenize(prediction)]
    score = 0
    for c_cent in tok_pred:
        try:
            score += corpus_bleu(
                [tok_ref], [c_cent], smoothing_function=smoothing_function
            )
        except KeyError:
            score = 0.0
    try:
        score /= len(tok_pred)
    except ZeroDivisionError:
        score = 0.0

    return score


def get_bart_scores(scorer, preds, refs, texts, directions=("hr", "fa"), batch_size=4):
    scores = {}
    # There is no need to calculate how close we are to a metric!
    if "sh" in directions:
        scores["BARTScore-sh"] = np.array(
            scorer.score(texts, preds, batch_size=batch_size)
        )
    if "fa" in directions or "rh" in directions:
        scores["BARTScore-rh"] = np.array(
            scorer.score(refs, preds, batch_size=batch_size)
        )
    if "fa" in directions or "hr" in directions:
        scores["BARTScore-hr"] = np.array(
            scorer.score(preds, refs, batch_size=batch_size)
        )
    if "fa" in directions:
        scores["BARTScore-fa"] = (scores["BARTScore-rh"] + scores["BARTScore-hr"]) / 2
    return scores
