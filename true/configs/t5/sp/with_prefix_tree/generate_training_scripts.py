template_seed = 42

base_seeds = [42, 23419, 4837, 705525, 12345]

for base_seed in base_seeds:
    for seed_incr in range(5):
        seed = base_seed + seed_incr

        with open(f'train_t5_large_rubq_{template_seed}.yaml', 'r') as handle:
            content = handle.read()
            new_content = content.replace(str(template_seed), str(seed))

            with open(f'train_t5_large_rubq_{seed}.yaml', 'w') as output_handle:
                output_handle.write(new_content)
