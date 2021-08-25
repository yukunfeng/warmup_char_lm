##  Warming Up Character-aware NLMs
Pytorch Implementation of RANLP 2021 paper [Improving Character-Aware Neural
Language Modelby Warming Up Character Encoder under Skip-gram
Architecture](to_do)
 
## Requirements
 - Python version >= 3.5
 - Pytorch version 0.4.0

## Datasets
Originally downloaded from [here](http://people.ds.cam.ac.uk/dsg40/lmmrl.html)
from [this
paper](https://www.aclweb.org/anthology/Q18-1032.pdf). Currently the link seems
broken and I have
uploaded one English dataset for testing under 'data' directory. 

## Usage
Run character-aware NLMs:

```
python -u main.py --dropout 0.5  --input_freq 1 \
    --max_gram_n 3 --note "char_lm"
    --data ./data/en/ --epoch 40 \
    --emsize 650 --nhid 650
```

Run character-aware NLMs with warmed up character encoder:
```
python -u main.py --dropout 0.5 --use_word2vec --input_freq 1 \
    --max_gram_n 3 --note "char_lm_warmup" \
    --data ./data/en/ --epoch 40 --emsize 650 --nhid 650
```
