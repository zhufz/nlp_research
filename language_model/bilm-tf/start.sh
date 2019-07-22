#git :https://github.com/allenai/bilm-tf

#setup:
python setup.py install

# prepare vocab file
python3 create_vocab.py  data/corpus.txt

# train
python3 bin/train_elmo.py \
    --train_prefix='data/corpus.txt' \
    --vocab_file='data/vocab.txt'  \
    --save_dir='data/checkpoint'
