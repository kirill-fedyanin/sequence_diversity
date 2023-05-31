from datasets import Dataset
from numpy.random import default_rng
from transformers import AutoTokenizer

from ue_abssum.add_ood import corrupt, add_ood_from_dataset

NORMAL = Dataset.from_dict(
    {
        "text": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
        "label": list(range(12)),
    }
)


OOD = Dataset.from_dict(
    {
        "text": [
            "this is ood text",
            "this is another ood text",
            "this is final ood text",
        ],
        "label": [1, 2, 3],
    }
)


TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")


def test_corrupt():
    text = "corrupt this text!"
    result = corrupt(text, TOKENIZER, default_rng(seed=42))
    assert result != text, "corrupt does not change text"


def get_match_count(dataset):
    matches = 0
    for row in OOD:
        matches += row["text"] in dataset["text"]
    return matches


def test_add_ood_from_dataset():
    gen = default_rng(seed=123)
    dataset_with_ood = add_ood_from_dataset(NORMAL, OOD, gen, is_ood_label_field="ood", shuffle=False)
    assert len(dataset_with_ood) == len(NORMAL) + 3
    assert get_match_count(dataset_with_ood[-3:]) == 3

    shuffled_dataset_with_ood = add_ood_from_dataset(NORMAL, OOD, gen, is_ood_label_field="ood", shuffle=True)
    assert len(shuffled_dataset_with_ood) == len(NORMAL) + 3
    assert get_match_count(shuffled_dataset_with_ood[-3:]) != 3
    assert get_match_count(shuffled_dataset_with_ood) == 3

    corrupted_dataset_with_ood = add_ood_from_dataset(
        NORMAL, OOD, gen, is_ood_label_field="ood", corrupt_samples=True, tokenizer=TOKENIZER
    )
    assert len(corrupted_dataset_with_ood) == len(NORMAL) + 3
    assert get_match_count(corrupted_dataset_with_ood) == 0
