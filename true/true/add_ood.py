from datasets import Dataset, concatenate_datasets

from numpy.random import Generator


def corrupt(text: str, tokenizer, gen: Generator) -> str:
    assert tokenizer is not None, "Tokenizer is not defined"
    tokens = tokenizer.encode(text)
    tokens = tokens[1:-1]  # Remove SEP and CLS tokens
    gen.shuffle(tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def corrupt_row(row, text_field, tokenizer, gen: Generator) -> dict:
    row[text_field] = corrupt(row[text_field], tokenizer, gen)
    return row


def add_ood_from_dataset(
    dataset: Dataset,
    ood: Dataset,
    random_gen: Generator,
    corrupt_samples: bool = False,
    text_field: str = "text",
    ood_text_field: str = "text",
    is_ood_label_field: str = "label",  # add column, where 1 is ood, 0 otherwise, or do nothing if None
    label_field: str = None,
    ood_label_field: str = None,
    shuffle: bool = False,
    remove_other_columns: bool = True,
    tokenizer=None,
) -> Dataset:
    if len(ood) > len(dataset):
        ood = ood.select(range(len(dataset)))
    if is_ood_label_field is not None:
        dataset = dataset.add_column(is_ood_label_field, [0] * len(dataset))
        ood = ood.add_column(is_ood_label_field, [1] * len(ood))

    if text_field != ood_text_field:
        ood = ood.rename_column(ood_text_field, text_field)
    if (
        label_field is not None
        and ood_label_field is not None
        and label_field != ood_label_field
    ):
        ood = ood.rename_column(ood_label_field, label_field)

    if corrupt_samples:
        ood = ood.map(
            lambda row: corrupt_row(row, text_field, tokenizer, random_gen),
            load_from_cache_file=False,
        )

    result = concatenate_datasets([dataset, ood])
    if remove_other_columns:
        result = result.remove_columns(
            [
                col
                for col in dataset.column_names
                if col not in [text_field, is_ood_label_field, label_field]
            ]
        )
    if shuffle:
        result = result.shuffle(random_gen)
    return result
