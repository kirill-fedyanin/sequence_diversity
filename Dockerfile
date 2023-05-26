FROM determinedai/environments:cuda-11.3-pytorch-1.10-deepspeed-0.8.3-gpu-0.22.1
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
RUN pip install transformers
RUN pip install -U sentence-transformers
RUN pip install tqdm numpy scipy matplotlib scikit-learn tensorboardX ipdb
