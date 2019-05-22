import tensorflow as tf
import numpy as np
import pdb


class Transformer():
    def __init__(self, **args):
        self.maxlen = args['maxlen']
        self.d_model = args['embedding_size']
        self.dropout_rate = 1 - args['keep_prob']
        self.num_output = args['num_output']
        self.training = args['is_training']
        self.num_blocks = 3
        self.num_heads = 8
        self.d_ff = 512 #hidden size


    def multihead_attention(self,
                            queries, keys, values,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_attention",
                            reuse = tf.AUTO_REUSE):
        d_model = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=reuse):
            # Linear projections
            Q = tf.layers.dense(queries, d_model, use_bias=False) # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=False) # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=False) # (N, T_k, d_model)
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, causality,
                                                        dropout_rate, training,
                                                        reuse = tf.AUTO_REUSE)
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
            # Residual connection
            outputs += queries
            # Normalize
            outputs = self.ln(outputs)
        return outputs

    def scaled_dot_product_attention(self,
                                     Q, K, V,
                                     causality=False, dropout_rate=0.,
                                     training=True,
                                     scope="scaled_dot_product_attention",
                                     reuse = tf.AUTO_REUSE):
        '''See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            # key masking
            outputs = self.mask(outputs, Q, K, type="key")

            # causality or future blinding masking
            if causality:
                outputs = self.mask(outputs, type="future")

            # softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # query masking
            outputs = self.mask(outputs, Q, K, type="query")

            # dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def mask(self, inputs, queries=None, keys=None, type=None):
        """Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (N, T_q, T_k)
        queries: 3d tensor. (N, T_q, d)
        keys: 3d tensor. (N, T_k, d)
        e.g.,
        >> queries = tf.constant([[[1.],
                            [2.],
                            [0.]]], tf.float32) # (1, 3, 1)
        >> keys = tf.constant([[[4.],
                         [0.]]], tf.float32)  # (1, 2, 1)
        >> inputs = tf.constant([[[4., 0.],
                                   [8., 0.],
                                   [0., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "key")
        array([[[ 4.0000000e+00, -4.2949673e+09],
            [ 8.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
        >> inputs = tf.constant([[[1., 0.],
                                 [1., 0.],
                                  [1., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "query")
        array([[[1., 0.],
            [1., 0.],
            [0., 0.]]], dtype=float32)
        """
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
            masks = tf.expand_dims(masks, 1) # (N, 1, T_k)
            masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

            # Apply masks to inputs
            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
        elif type in ("q", "query", "queries"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
            masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
            masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

            # Apply masks to inputs
            outputs = inputs*masks
        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

            paddings = tf.ones_like(masks) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")
        return outputs


    def ff(self, inputs, num_units, scope="positionwise_feedforward", \
           reuse = tf.AUTO_REUSE):
        '''position-wise feed forward net. See 3.3
        inputs: A 3d tensor with shape of [N, T, C].
        num_units: A list of two integers.
        scope: Optional scope for `variable_scope`.
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])
            # Residual connection
            outputs += inputs
            # Normalize
            outputs = self.ln(outputs, reuse = reuse)
        return outputs

    def ln(self, inputs, epsilon = 1e-8, scope="ln", reuse = tf.AUTO_REUSE):
        '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
        return outputs
    def positional_encoding(self,
                            inputs,
                            maxlen,
                            masking=True,
                            scope="positional_encoding",
                            reuse = tf.AUTO_REUSE):
        '''Sinusoidal Positional_Encoding. See 3.5
        inputs: 3d tensor. (N, T, E)
        maxlen: scalar. Must be >= T
        masking: Boolean. If True, padding positions are set to zeros.
        scope: Optional scope for `variable_scope`.
        returns
        3d tensor that has the same shape as inputs.
        '''

        E = inputs.get_shape().as_list()[-1] # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
        with tf.variable_scope(scope, reuse=reuse):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

    def feed_dict(self, **kwargs):
        feed_dict = {}
        return feed_dict

    def pb_feed_dict(self, graph, **kwargs):
        feed_dict = {}
        return feed_dict

    def __call__(self, enc, name = 'encoder', reuse = tf.AUTO_REUSE, **kwargs):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        #pdb.set_trace()
        with tf.variable_scope("transformer", reuse = reuse):
            #enc += self.positional_encoding(enc, self.maxlen)
            #enc = tf.layers.dropout(enc, dropout_rate, training=self.training)

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = self.multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=self.training,
                                              causality=False,
                                              reuse=reuse)
                    # feed forward
                    enc = self.ff(enc, num_units=[self.d_ff, self.d_model],
                                  reuse=reuse)
            memory = tf.reshape(enc, [-1, self.maxlen*self.d_model])
            dense = tf.layers.dense(memory, self.num_output, activation=None)
        return dense


