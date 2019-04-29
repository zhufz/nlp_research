import keras
import tensorflow as tf
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model

#refer:https://github.com/EternalFeather/ESIM

class ESIM():
    def __init__(self, **kwargs):
        self.maxlen = kwargs['maxlen']
        self.embedding_size = kwargs['embedding_size']
        self.keep_prob = args['keep_prob']
        self.num_output = args['num_output']
        self.recurrent_units = 300
        self.dense_units = 300
        self.dropout_rate = 1- self.keep_prob

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self,graph, name = 'esim',  **kwargs):
        feed_dict = {}
        return feed_dict

    def __call__(self, x_query, x_sample, reuse = tf.AUTO_REUSE):
        embedding_sequence_q1 = BatchNormalization(axis=2)(x_query)
        embedding_sequence_q2 = BatchNormalization(axis=2)(x_sample)

        final_embedding_sequence_q1 = SpatialDropout1D(0.25)(embedding_sequence_q1)
        final_embedding_sequence_q2 = SpatialDropout1D(0.25)(embedding_sequence_q2)

        rnn_layer_q1 = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(final_embedding_sequence_q1)
        rnn_layer_q2 = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(final_embedding_sequence_q2)

        attention = Dot(axes=-1)([rnn_layer_q1, rnn_layer_q2])
        w_attn_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
        w_attn_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
        align_layer_1 = Dot(axes=1)([w_attn_1, rnn_layer_q1])
        align_layer_2 = Dot(axes=1)([w_attn_2, rnn_layer_q2])

        subtract_layer_1 = subtract([rnn_layer_q1, align_layer_1])
        subtract_layer_2 = subtract([rnn_layer_q2, align_layer_2])

        multiply_layer_1 = multiply([rnn_layer_q1, align_layer_1])
        multiply_layer_2 = multiply([rnn_layer_q2, align_layer_2])

        m_q1 = concatenate([rnn_layer_q1, align_layer_1, subtract_layer_1, multiply_layer_1])
        m_q2 = concatenate([rnn_layer_q2, align_layer_2, subtract_layer_2, multiply_layer_2])

        v_q1_i = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(m_q1)
        v_q2_i = Bidirectional(LSTM(self.recurrent_units, return_sequences=True))(m_q2)

        avgpool_q1 = GlobalAveragePooling1D()(v_q1_i)
        avgpool_q2 = GlobalAveragePooling1D()(v_q2_i)
        maxpool_q1 = GlobalMaxPooling1D()(v_q1_i)
        maxpool_q2 = GlobalMaxPooling1D()(v_q2_i)

        merged_q1 = concatenate([avgpool_q1, maxpool_q1])
        merged_q2 = concatenate([avgpool_q2, maxpool_q2])

        final_v = BatchNormalization()(concatenate([merged_q1, merged_q2]))
        output = Dense(units=self.dense_units, activation='relu')(final_v)
        output = BatchNormalization()(output)
        output = Dropout(self.dropout_rate)(output)
        output = Dense(units=self.num_output, activation='sigmoid')(output)
        output = tf.squeeze(output, -1)
        return output

