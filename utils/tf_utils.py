import sys,os
import tensorflow as tf
from tensorflow.python.framework import graph_util

def load_pb(path_to_model):
    if not os.path.exists(path_to_model):
        raise ValueError("'path_to_model' is not exist.")
    model_graph = tf.Graph()
    #model_graph = tf.get_default_graph()  error
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return model_graph


def write_pb(checkpoint_path, pb_path, output_nodes):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    sess = tf.Session()

    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               input_graph_def,# 等于:sess.graph_def
                                                               output_nodes)
    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile(pb_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())


if __name__ == '__main__':
    write_pb(checkpoint_path = "./data/classify", pb_path \
          ="./data/model/model.pb", output_nodes = ['output/scores','output/predictions','accuracy/accuracy'])
