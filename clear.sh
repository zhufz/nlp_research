find ./ -name __pycache__|xargs rm -rf
find ./ -name *.pyc|xargs rm
find ./ -name .DS_Store|xargs rm -rf

rm -rf ./logs
rm -rf ./data/classify/checkpoint
rm -rf ./data/classify/model.pb
rm -rf ./data/ner/checkpoint
rm -rf ./data/ner/model.pb
rm -rf ./data/match/checkpoint
rm -rf ./data/match/model.pb


