#!/usr/bin/env bash

set -x

data_path="./data/zh"

python -u main.py --dropout 0.5  --input_freq 1 --max_gram_n 3 --note "char_lm" \
    --data $data_path --epoch 40 --emsize 650 --nhid 650  >> log.txt

python -u main.py --dropout 0.5 --use_word2vec --input_freq 1 --max_gram_n 3 --note "char_lm_warmup" \
    --data $data_path --epoch 40 --emsize 650 --nhid 650  >> log.txt
