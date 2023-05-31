FROM determinedai/environments:cuda-11.3-pytorch-1.10-deepspeed-0.8.3-gpu-0.22.1
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
RUN pip install jupyter tqdm numpy scipy matplotlib scikit-learn tensorboardX pandas plotly
RUN pip install transformers==4.20.1 sentence-transformers datasets==2.3.2
RUN pip install levenshtein hydra-core omegaconf marisa-trie pytreebank wget peft rouge-score
RUN pip install nlpaug
