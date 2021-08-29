##  Warming Up Character-aware NLMs
Pytorch Implementation of RANLP 2021 paper [Improving Character-Aware Neural
Language Model by Warming Up Character Encoder under Skip-gram
Architecture](to_do)
 
## Requirements
 - Python version >= 3.5
 - Pytorch version 0.4.0

## Datasets
We used a subset of [lmmrl](http://people.ds.cam.ac.uk/dsg40/lmmrl.html) datasets containing 50 different languages ([Gerz et al., 2018](https://www.aclweb.org/anthology/Q18-1032.pdf)). Currently the download link seems broken and I have uploaded one English dataset for testing under 'data' directory.

## Usage
Run character-aware NLMs:

```
python -u main.py --dropout 0.5  --input_freq 1 --max_gram_n 3 --note "char_lm" \
    --data ./data/en/ --epoch 40 --emsize 650 --nhid 650
```
`max_gram_n` is the character gram used in the paper and is set to 1 for Japanese and Chinese and 3 for other languages.

Run the following command to see the details.
```
python ./main.py -h
```

Run character-aware NLMs with warmed up character encoder:
```
python -u main.py --dropout 0.5 --use_word2vec --input_freq 1 --max_gram_n 3 --note "char_lm_warmup" \
    --data ./data/en/ --epoch 40 --emsize 650 --nhid 650
```
Note that the Skip-gram implementation in the code is not optimized and thus the training is quite slow. The configuration for skip-gram is fixed in the code, such as epoch, window size, batch_size etc. More details are in `word2vec.py`).


