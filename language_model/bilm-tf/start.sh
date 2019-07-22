#git :https://github.com/allenai/bilm-tf
#other:https://github.com/nefujiangping/entity_recognition/blob/master/README.md#how-to-train-a-pure-token-level-elmo-from-scratch

#setup:
#python setup.py install

# prepare vocab file
#python3 create_vocab.py  data/corpus.txt

# train
python3 bin/train_elmo.py \
    --train_prefix='data/corpus.txt' \
    --vocab_file='data/vocab.txt'  \
    --save_dir='data/checkpoint'
