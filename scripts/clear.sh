find ./ -name *__pycache__*|xargs rm -rf
find ./ -name *.pyc|xargs rm
find ./ -name .DS_Store|xargs rm -rf

rm -rf ./logs
#find ./ -name checkpoint|xargs rm -rf
#find ./ -name tfrecords|xargs rm -rf


