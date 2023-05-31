import os


MAX_TOKENIZER_LENGTH = (
    16 if os.environ.get("TESTING") else None
)  # FIXME remove this constant


def tokenize_data(
    data,
    tokenizer,
    document_name="document",
    label_name="",
    batched=True,
    padding=False,
):
    def tokenize_fn(instances):
        encoded = tokenizer(
            instances[document_name],
            truncation=True,
            padding=padding,
            max_length=MAX_TOKENIZER_LENGTH,
        )
        if label_name in instances:
            with tokenizer.as_target_tokenizer():
                if isinstance(instances[label_name], list):
                    labels = []
                    for instance_label_names in instances[label_name]:
                        instance_labels = tokenizer(
                            instance_label_names,
                            truncation=True,
                            padding=padding,
                            max_length=MAX_TOKENIZER_LENGTH,
                        )
                        labels.append(instance_labels["input_ids"])
                    encoded["labels"] = labels
                else:
                    labels = tokenizer(
                        instances[label_name],
                        truncation=True,
                        padding=padding,
                        max_length=MAX_TOKENIZER_LENGTH,
                    )

                    encoded["labels"] = labels["input_ids"]
        return encoded

    columns_to_remove = [x for x in data.features.keys() if x != "labels"]
    if os.path.exists("tmp.data"):
        # It may have already been deleted at this time
        try:
            os.remove("tmp.data")
        except Exception:
            pass
    return data.map(
        tokenize_fn,
        batched=batched,
        remove_columns=columns_to_remove,
        load_from_cache_file=False,
        cache_file_name="tmp.data",
    )
