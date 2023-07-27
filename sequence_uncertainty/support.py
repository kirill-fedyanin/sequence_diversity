from random import randint
APPROX_LETTERS_PER_TOKEN = 7

def sample_sequences(tokenizer, dataset, ctx_size, num_samples):
    max_attempts = 10_000
    tokens = []
    for _ in range(num_samples):
        for _ in range(max_attempts):
            i = randint(0, len(dataset))
            text = dataset[i]['text']
            if len(text) < ctx_size * APPROX_LETTERS_PER_TOKEN:
                continue
            else:
                tokenized = tokenizer(text[:ctx_size*APPROX_LETTERS_PER_TOKEN], return_tensors='pt').input_ids
                tokens.append(tokenized[0, :ctx_size])
                break
    return tokens

