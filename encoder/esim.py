import keras
import tensorflow as tf
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model
from encoder import Base

#refer:https://arxiv.org/abs/1609.06038

class ESIM(Base):
    def __init__(self, **kwargs):
        super(ESIM, self).__init__(**kwargs)
        self.maxlen = kwargs['maxlen']
        self.embedding_size = kwargs['embedding_size']
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']
        self.recurrent_units = 300
        self.dense_units = 300

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self,graph, name = 'esim',  **kwargs):
        feed_dict = {}
        return feed_dict

    def update_features(self, features):
        pass

    def __call__(self, x_query, x_sample, reuse = tf.AUTO_REUSE, **kwargs):
        #embedding_sequence_q1 = BatchNormalization(axis=2)(x_query)
        #embedding_sequence_q2 = BatchNormalization(axis=2)(x_sample)
        #final_embedding_sequence_q1 = SpatialDropout1D(0.25)(embedding_sequence_q1)
        #final_embedding_sequence_q2 = SpatialDropout1D(0.25)(embedding_sequence_q2)

        #################### 输入编码input encoding #######################
        #分别对query和sample进行双向编码
        rnn_layer_q1 = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(x_query)
        rnn_layer_q2 = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(x_sample)
        #rnn_layer_q1 = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(final_embedding_sequence_q1)
        #rnn_layer_q2 = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(final_embedding_sequence_q2)

        ############## 局部推理local inference modeling ###################
        #计算dot attention
        attention = Dot(axes=-1)([rnn_layer_q1, rnn_layer_q2])
        #分别计算query和sample进行attention后的结果
        w_attn_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
        w_attn_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
        align_layer_1 = Dot(axes=1)([w_attn_1, rnn_layer_q1])
        align_layer_2 = Dot(axes=1)([w_attn_2, rnn_layer_q2])

        ############# 推理组合Inference Composition #######################

        subtract_layer_1 = subtract([rnn_layer_q1, align_layer_1])
        subtract_layer_2 = subtract([rnn_layer_q2, align_layer_2])

        multiply_layer_1 = multiply([rnn_layer_q1, align_layer_1])
        multiply_layer_2 = multiply([rnn_layer_q2, align_layer_2])

        m_q1 = concatenate([rnn_layer_q1, align_layer_1, subtract_layer_1, multiply_layer_1])
        m_q2 = concatenate([rnn_layer_q2, align_layer_2, subtract_layer_2, multiply_layer_2])

        ############### 编码+池化 #######################
        v_q1_i = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(m_q1)
        v_q2_i = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(m_q2)

        avgpool_q1 = GlobalAveragePooling1D()(v_q1_i)
        avgpool_q2 = GlobalAveragePooling1D()(v_q2_i)
        maxpool_q1 = GlobalMaxPooling1D()(v_q1_i)
        maxpool_q2 = GlobalMaxPooling1D()(v_q2_i)

        merged_q1 = concatenate([avgpool_q1, maxpool_q1])
        merged_q2 = concatenate([avgpool_q2, maxpool_q2])

        final_v = BatchNormalization()(concatenate([merged_q1, merged_q2]))
        #output = Dense(units=self.dense_units, activation='relu')(final_v)
        output = Dense(units=self.num_output, activation=None)(final_v)
        #output = BatchNormalization()(output)
        #output = Dropout(self.dropout_rate)(output)
        #output = tf.nn.dropout(output, self.keep_prob)
        #高级api tf.layer.dropout 与 keras的Dropout都使用dropout
        #tf.nn.dropout使用keep_prob
        #output = Dense(units=self.num_output, activation='sigmoid')(output)
        #output = Dense(units=self.num_output, activation=None)(output)
        output = tf.squeeze(output, -1)
        return output

