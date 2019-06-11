#-*- coding:utf-8 -*-
import tensorflow as tf
from encoder import EncoderBase



class HAN(EncoderBase):
    def __init__(self, **kwargs):
        super(HAN, self).__init__(**kwargs)
        self.seq_length = kwargs['maxlen']
        self.embedding_dim = kwargs['embedding_size']
        self.num_sentences = 10
        self.hidden_dim = 128
        self.context_dim = 256
        self.rnn_type = "lstm"
        self.keep_prob = kwargs['keep_prob']


    def __call__(self, embed, name = 'encoder', reuse = tf.AUTO_REUSE, **kwargs):

        def _get_cell():
            if self.rnn_type == "vanilla":
                return tf.nn.rnn_cell.BasicRNNCell(self.context_dim)
            elif self.rnn_type == "lstm":
                return tf.nn.rnn_cell.BasicLSTMCell(self.context_dim)
            else:
                return tf.nn.rnn_cell.GRUCell(self.context_dim)

        def _Bidirectional_Encoder(inputs, name):
            with tf.variable_scope(name, reuse = reuse):
                fw_cell = _get_cell()
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
                bw_cell = _get_cell()
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
                (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                 cell_bw=bw_cell,
                                                                                 inputs=inputs,
                                                                                 dtype=tf.float32)
            return output_fw, output_bw

        def _attention(inputs, name):
            with tf.variable_scope(name, reuse = reuse):
                # 使用一个全连接层编码 GRU 的输出，相当于一个隐藏层
                # [batch_size,sentence_length,hidden_size * 2]
                hidden_vec = tf.layers.dense(inputs, self.hidden_dim * 2,
                                             activation=tf.nn.tanh, name='w_hidden')

                # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
                # [hidden_size * 2]
                u_context = tf.Variable(tf.truncated_normal([self.hidden_dim * 2]), name='u_context')
                # [batch_size,sequence_length]
                alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_vec, u_context),
                                                    axis=2, keep_dims=True), dim=1)
                # before reduce_sum [batch_size, sequence_length, hidden_szie*2]，
                # after reduce_sum [batch_size, hidden_size*2]
                attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)

            return attention_output

        sentence_len = int(self.seq_length / self.num_sentences)
        embedding_inputs_reshaped = tf.reshape(embed,
                                                   shape=[-1, sentence_len, self.embedding_dim])
        with tf.variable_scope("word_vec", reuse = reuse):
            (output_fw, output_bw) = _Bidirectional_Encoder(embedding_inputs_reshaped, "word_vec")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            word_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.variable_scope("word_attention",reuse = reuse):
            """
           attention process:
           1.get logits for each word in the sentence.
           2.get possibility distribution for each word in the sentence.
           3.get weighted sum for the sentence as sentence representation.
           """
            # [batch_size*num_sentences, hidden_size * 2]
            sentence_vec = _attention(word_hidden_state, "word_attention")

        with tf.variable_scope("sentence_vec",reuse = reuse):
            # [batch_size,num_sentences,hidden_size*2]
            sentence_vec = tf.reshape(sentence_vec, shape=[-1, self.num_sentences,
                                                           self.context_dim * 2])
            output_fw, output_bw = _Bidirectional_Encoder(sentence_vec, "sentence_vec")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            sentence_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.variable_scope("sentence_attention", reuse = reuse):
            # [batch_size, hidden_size * 2]
            doc_vec = _attention(sentence_hidden_state, "sentence_attention")

        return doc_vec

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict
