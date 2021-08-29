#!/usr/bin/env bash

set -x

data_path="./data/zh"

# python -u main.py --dropout 0.5  --input_freq 1 --max_gram_n 3 --note "char_lm" \
    # --data $data_path --epoch 40 --emsize 650 --nhid 650

python -u main.py --dropout 0.5 --input_freq 1 --max_gram_n 3 --note "char_lm_warmup" \
    --data $data_path --epoch 40 --emsize 650 --nhid 650 --use_warmup \
    --skipgram_batch_size 200 --skipgram_window_size 5 --skipgram_epoch 7 --skipgram_lr 10
