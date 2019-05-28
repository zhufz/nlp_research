import tensorflow as tf
from language_model.bert import modeling
from language_model.bert import optimization
from language_model.bert import tokenization
from encoder import Base
import pdb
import copy

class Bert(Base):
    def __init__(self, **kwargs):
        """
        :param config:
        """
        super(Bert, self).__init__(**kwargs)
        self.maxlen = kwargs['maxlen']
        self.embedding_dim = kwargs['embedding_size']
        self.keep_prob = kwargs['keep_prob']
        self.num_output = kwargs['num_output']
        self.is_training = kwargs['is_training']
        self.bert_config_file = kwargs['bert_config_file_path']
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        self.vocab_file = kwargs['vocab_file_path']
        self.placeholder = {}

    def __call__(self, name = 'encoder', features = None, reuse = tf.AUTO_REUSE, **kwargs):
        self.placeholder[name+'_input_ids'] = tf.placeholder(tf.int32, 
                                        shape=[None, self.maxlen], 
                                        name = name+"_input_ids")
        self.placeholder[name+'_input_mask'] = tf.placeholder(tf.int32, 
                                        shape=[None, self.maxlen], 
                                        name = name+"_input_mask")
        self.placeholder[name+'_segment_ids'] = tf.placeholder(tf.int32, 
                                        shape=[None, self.maxlen], 
                                        name = name+"_segment_ids")
        if features != None:
            self.placeholder[name+'_input_ids'] = features[name+'_input_ids']
            self.placeholder[name+'_input_mask'] = features[name+'_input_mask']
            self.placeholder[name+'_segment_ids'] = features[name+'_segment_ids']

        with tf.variable_scope("bert", reuse = reuse):

            model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,#True,
                input_ids=self.placeholder[name+"_input_ids"],
                input_mask=self.placeholder[name+'_input_mask'],
                token_type_ids=self.placeholder[name+'_segment_ids'],
                use_one_hot_embeddings=False)

            output_layer = model.get_pooled_output()

            hidden_size = output_layer.shape[-1].value

            output_weights = tf.get_variable(
                "output_weights", [self.num_output, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias = tf.get_variable(
                "output_bias", [self.num_output], initializer=tf.zeros_initializer())

            with tf.variable_scope("loss"):
                output_layer = tf.nn.dropout(output_layer, keep_prob=self.keep_prob)
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                return logits

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def build_ids(self, text_a, text_b = None, **kwargs):
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, 
                                               do_lower_case=True)

        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if text_b:
          tokens_b = tokenizer.tokenize(text_b)
        if tokens_b:
          # Modifies `tokens_a` and `tokens_b` in place so that the total
          # length is less than the specified length.
          # Account for [CLS], [SEP], [SEP] with "- 3"
          self._truncate_seq_pair(tokens_a, tokens_b, self.maxlen - 3)
        else:
          # Account for [CLS] and [SEP] with "- 2"
          if len(tokens_a) > self.maxlen - 2:
            tokens_a = tokens_a[0:(self.maxlen - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.maxlen:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        #pdb.set_trace()

        assert len(input_ids) == self.maxlen
        assert len(input_mask) == self.maxlen
        assert len(segment_ids) == self.maxlen
        return input_ids, input_mask, segment_ids


    def feed_dict(self,text_a_list, text_b_list = None, name = 'encoder', **kwargs):
        feed_dict = {}
        input_ids_list, input_mask_list, segment_ids_list = [],[],[]
        for idx,text in enumerate(text_a_list):
            input_ids, input_mask, segment_ids = \
                self.build_ids(text, text_b_list[idx] if text_b_list != None else None)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
        feed_dict[self.placeholder[name+"_input_ids"]] = input_ids_list
        feed_dict[self.placeholder[name+'_input_mask']] = input_mask_list
        feed_dict[self.placeholder[name+'_segment_ids']] = segment_ids_list
        return feed_dict

    def pb_feed_dict(self, graph, text_a_list, text_b_list = None, name = 'encoder', **kwargs):
        feed_dict = {}
        input_ids_node = graph.get_operation_by_name(name+'_input_ids').outputs[0]
        input_mask_node = graph.get_operation_by_name(name+'_input_mask').outputs[0]
        segment_ids_node = graph.get_operation_by_name(name+'_segment_ids').outputs[0]

        input_ids_list, input_mask_list, segment_ids_list = [],[],[]
        for idx,text in enumerate(text_a_list):
            input_ids, input_mask, segment_ids = \
                self.build_ids(text, text_b_list[idx] if text_b_list != None else None)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
        feed_dict[input_ids_node ] = input_ids_list
        feed_dict[input_mask_node ] = input_mask_list
        feed_dict[segment_ids_node ] = segment_ids_list
        return feed_dict

    def encoder_fun(self, raw, name = 'encoder', **kwargs):
        input_ids, input_mask, segment_ids = \
            self.build_ids(x_query_raw, None)
        return {name+"_input_ids": input_ids, 
                name+"_input_mask": input_mask, 
                name+"_segment_ids": segment_ids}

    def keys_to_features(self, name = 'encoder'):
        keys_to_features = {
            name+"_input_ids": tf.FixedLenFeature([self.maxlen], tf.int64), 
            name+"_input_mask": tf.FixedLenFeature([self.maxlen], tf.int64), 
            name+"_segment_ids": tf.FixedLenFeature([self.maxlen], tf.int64)
        }
        return keys_to_features

    def parsed_to_features(self, parsed, name = 'encoder'):
        ret = {
            name + "_input_ids": tf.reshape(parsed[name+ "_input_ids"], [self.maxlen]), 
            name + "_input_mask": tf.reshape(parsed[name + "_input_ids"], [self.maxlen]),
            name+"_segment_ids": tf.reshape(parsed[name + "_input_ids"], [self.maxlen])
        }
        return ret

    def get_features(self, name = 'encoder'):
        features = {}
        features[name+'_input_ids'] = tf.placeholder(tf.int32, 
                                        shape=[None, self.maxlen], 
                                        name = name+"_input_ids")
        features[name+'_input_mask'] = tf.placeholder(tf.int32, 
                                        shape=[None, self.maxlen], 
                                        name = name+"_input_mask")
        features[name+'_segment_ids'] = tf.placeholder(tf.int32, 
                                        shape=[None, self.maxlen], 
                                        name = name+"_segment_ids")
        return features
