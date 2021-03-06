##  Warming Up Character-aware NLMs
Pytorch Implementation of RANLP 2021 paper ["Improving Character-Aware Neural
Language Model by Warming Up Character Encoder under Skip-gram
Architecture"](to_do)
 
## Requirements
 - Python version >= 3.5
 - Pytorch version 0.4.0

## Datasets
We used a subset of [lmmrl](http://people.ds.cam.ac.uk/dsg40/lmmrl.html) datasets containing 50 different languages ([Gerz et al., 2018](https://www.aclweb.org/anthology/Q18-1032.pdf)). Currently the download link seems broken and I have uploaded one English dataset for testing under 'data' directory.

## Usage
```
python main.py -h

   -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --skip_gram_use_word  whether add word in skipgram model
  --seed SEED           random seed
  --device DEVICE       cuda
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --freq_thre FREQ_THRE
                        freq threshould
  --max_gram_n MAX_GRAM_N
                        character n-gram. We use 1 for japanese and Chinese
                        and 3 for other languages.
  --input_freq INPUT_FREQ
                        freq threshould for input word
  --note NOTE           extra note in final one-line result output
  --use_warmup          whether warm up character encoder
  --skipgram_batch_size SKIPGRAM_BATCH_SIZE
                        batch size for skipgram if warmup is used
  --skipgram_window_size SKIPGRAM_WINDOW_SIZE
                        window size for skipgram if warmup is used
  --skipgram_epoch SKIPGRAM_EPOCH
                        training epoch of skipgram if warmup is used
  --skipgram_lr SKIPGRAM_LR
                        learning rate for skipgram if warmup is used.
```

## Reproduce results in the paper
Run character-aware NLMs:

```
python -u main.py --dropout 0.5  --input_freq 1 --max_gram_n 3 --note "char_lm" \
    --data ./data/en/ --epoch 40 --emsize 650 --nhid 650
```

Run character-aware NLMs with warmed up character encoder:
```
python -u main.py --dropout 0.5 --input_freq 1 --max_gram_n 3 --note "char_lm_warmup" \
    --data $data_path --epoch 40 --emsize 650 --nhid 650 --use_warmup \
    --skipgram_batch_size 200 --skipgram_window_size 5 --skipgram_epoch 7 --skipgram_lr 10
```
The skip-gram implementation (adapted from [skip-gram-pytorch](https://github.com/fanglanting/skip-gram-pytorch)) in the code is not optimized and thus the training is quite slow.
