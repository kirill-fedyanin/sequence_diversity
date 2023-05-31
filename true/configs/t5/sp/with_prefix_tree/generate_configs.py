import numpy as np
import os

topps = np.linspace(0.01, 1.0, num=20).round(2)

for topp in topps:
    command = f'cp sp_sq_en_single_large_ft_sample.yaml sp_sq_en_single_large_ft_sample_topp_{topp}.yaml' 
    os.system(command)
