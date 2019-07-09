import tensorflow as tf 
import logging
import numpy as np

class ESIM(object):
    def __init__(self, config):
        #logging.basicConfig(level=logging.DEBUG,
        #                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.config = config
        self.logger=logging.getLogger("SEMANTIC_MATCH")
        self.debug_dict = {}
        #build model
        
        #add place holder
        self.lr=config.lr
        self.lstm_dim=config.lstm_dim
        self.max_step=config.max_step
        self.train_drop_keep_prob=config.dropout_keep_prob
        self.embedding_size=config.embedding_size
     
        self.std_que=tf.placeholder(tf.int32,[None,None])
        self.cus_que=tf.placeholder(tf.int32,[None,None])
        self.labels = tf.placeholder(tf.int32,[None])
        self.batch_size=tf.shape(self.std_que)[0]
        self.dropout_keep_prob=tf.placeholder(tf.float32)
      
        self.std_que_len=tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.std_que)),reduction_indices=1),tf.int32)
        self.cus_que_len=tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.cus_que)),reduction_indices=1),tf.int32)
        self.std_mask = tf.sequence_mask(self.std_que_len, self.max_step) 
        self.cus_mask = tf.sequence_mask(self.cus_que_len, self.max_step) 
        self.global_step = tf.Variable(0, name="global_step")
        self.debug_dict['std_que'] = tf.shape(self.std_que)
        #word embedding
        self.std_emb = self.word_embedding_layer(self.std_que)
        self.cus_emb = self.word_embedding_layer(self.cus_que)
        self.debug_dict['std_emb'] = tf.shape(self.std_emb)
        #bilstm encode
        self.std_encoded = self.bilstm_layer(self.std_emb, self.std_que_len, scope = 'bilstm_encoder')
        self.cus_encoded = self.bilstm_layer(self.cus_emb, self.cus_que_len, scope = 'bilstm_encoder')
        #local inference
        self.debug_dict['std_encoded'] = tf.shape(self.std_encoded)
        self.std_m, self.cus_m = self.local_inference_layer(self.std_encoded, self.cus_encoded, std_mask = self.std_mask, cus_mask = self.cus_mask)
        #inference composition
        self.dense_dim = 256 #dense layer to avoid overfitting, see section 3.3 of the paper 
        self.std_p = self.dense_layer(self.std_m, scope = 'composition_fn')
        self.cus_p = self.dense_layer(self.cus_m, scope = 'composition_fn')

        self.std_decoded = self.bilstm_layer(self.std_p, seq_len = self.std_que_len, scope = 'bilstm_decoder')
        self.cus_decoded = self.bilstm_layer(self.cus_p, seq_len = self.cus_que_len, scope = 'bilstm_decoder')


        #pooling
        self.v = self.pooling_layer(self.std_decoded, self.cus_decoded, std_mask = self.std_mask, cus_mask = self.cus_mask)
        #final inference
        self.logits = self.dense_layer(self.v, scope = 'final_inference', hidden_dims = [128, 2], activations =[tf.nn.relu, tf.nn.tanh])
        #predictions
        self.predictions = tf.reshape(tf.cast(tf.argmax(self.logits, axis = 1), tf.int32),(-1,))
        #metrics
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
        #cost
        labels = tf.one_hot(self.labels, 2)
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=self.logits))
        #optimizer
        self.optimizer_layer()
        #summary
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        #saver
        self.saver=tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def word_embedding_layer(self, texts):
        '''
        :input  : texts [batch_size, seq]
        :return : embedding [batch_size, seq, embedding_size]
        '''
        with tf.variable_scope("word_embedding", reuse = tf.AUTO_REUSE):
            if False:#self._train_params['pretrain_flag']:
                initializer = np.load(self._train_params['pretrain_npy'])
                self.word_embedding=tf.get_variable(name="word_embedding",initializer=initializer)
                self.embedding_size = np.shape(initializer)[-1]
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
                self.word_embedding=tf.get_variable(name="word_embedding",shape=[self.config.vocab_size+1,self.embedding_size],initializer=initializer)
        embedding = tf.nn.embedding_lookup(self.word_embedding, texts)
        '''																																					                                     
        with tf.variable_scope("word_embedding", reuse = tf.AUTO_REUSE):
            initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
            word_embedding=tf.get_variable(
                        name="word_embedding",
                        shape=[self._train_params['vocab_size']+1,self._train_params['embedding_size']],
                        initializer=initializer
                        )
            embedding = tf.nn.embedding_lookup(word_embedding, texts)
        '''
        return embedding

    def bilstm_layer(self, seq, seq_len, scope = None):
        '''
        :input  : seq       [batch_size, seq_len, embedding_size]
                  seq_len   [batch_size, seq_len]
        :return : embedding [batch_size, seq_len, 2*lstm_hidden_dim]
        '''
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            lstm_forward = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
            lstm_backward = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dim)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_forward, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_backward, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, seq, dtype=tf.float32, sequence_length=seq_len)
        self.debug_dict[scope+'-outputs'] = tf.shape(tf.concat(outputs, axis=2))
        return tf.concat(outputs, axis=2)

    def local_inference_layer(self, std_bar, cus_bar, std_mask = None, cus_mask = None):
        '''
        :input  : std      [batch_size, seq_len, 2*lstm_hidden_dim]
                  cus      [batch_size, seq_len, 2*lstm_hidden_dim]
        :return : std_m    [batch_size, 4, 2*lstm_hidden_dim] ( 4 represents different combination according to paper)
                  cus_m    [batch_size, 4, 2*lstm_hidden_dim]
        '''
        e = tf.matmul(std_bar, tf.transpose(cus_bar, perm = [0, 2, 1]))
        self.debug_dict['std_bar'] = tf.shape(std_bar)
        def get_len(seq,axis=1):
            return tf.cast(tf.reduce_sum(tf.sign(tf.abs(seq)),reduction_indices=axis),tf.int32)

        if std_mask is None:
            attention_cus = tf.nn.softmax(e, axis = 2)
            attention_std = tf.nn.softmax(e, axis = 1)
        else:
            def mask_softmax(tensor, axis, mask):
                exp_tensor = tf.exp(tensor)
                masked_exp_tensor=tf.multiply(exp_tensor,tf.cast(mask,tf.float32))
                return masked_exp_tensor / tf.reduce_sum(masked_exp_tensor, axis, keepdims=True)
  
            cus_mask = tf.tile(tf.expand_dims(cus_mask,1), (1,self.max_step,1))
            attention_cus = mask_softmax(e, 2, cus_mask)
            std_mask = tf.tile(tf.expand_dims(std_mask,2), (1,1,self.max_step))
            attention_std = mask_softmax(e, 1, std_mask)

            attention_cus = tf.multiply(attention_cus,tf.cast(std_mask,tf.float32))
            attention_std = tf.multiply(attention_std,tf.cast(cus_mask,tf.float32))
        std_hat = tf.matmul(attention_cus, cus_bar)
        cus_hat = tf.matmul(tf.transpose(attention_std, perm = [0, 2, 1]), std_bar)
        std_diff = tf.math.subtract(std_bar, std_hat)
        cus_diff = tf.math.subtract(cus_bar, cus_hat)
        std_mul = tf.math.multiply(std_bar, std_hat)
        cus_mul = tf.math.multiply(cus_bar, cus_hat)
        std_m = tf.concat([std_bar, std_hat, std_diff, std_mul], axis = -1)
        cus_m = tf.concat([cus_bar, cus_hat, cus_diff, cus_mul], axis = -1)
        return std_m, cus_m

    def dense_layer(self, dense_input, hidden_dims = None, activations = None, scope = None):
        '''
        :input : denseinput  [batch_szie, hidden_dim_1]
        :return: output [batch_size, hidden_dim_n]
        **to be modified
        '''
        if hidden_dims == None:
            hidden_dims = [self.dense_dim]
            activations = [tf.nn.relu]
        o = dense_input
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE) as scope:
            for layer_dim, activation in zip(hidden_dims,activations):
                dropout = tf.layers.dropout(o, rate= 1 - self.dropout_keep_prob)
                o = tf.layers.dense(dropout, layer_dim, activation = activation)
        return o

    def pooling_layer(self, std_v, cus_v, std_mask = None, cus_mask = None):
        '''
        :input : std_v  [batch_size, seq_len, 2*lstm_hidden_dim]
                 cus_v  [batch_szie, seq_len, 2*lstm_hidden_dim]
        :return: v  [batch_size, 4*2*lstm_hidden_dim]
        '''
        '''
        if std_mask is not None:
            hdim = tf.shape(std_v)[-1]
            std_mask = tf.tile(tf.expand_dims(std_mask,2), (1,1,hdim))
            cus_mask = tf.tile(tf.expand_dims(cus_mask,2), (1,1,hdim))
            std_v = tf.boolean_mask(std_v,std_mask)
            _, std_v1 = tf.dynamic_partition(std_v, tf.cast(std_mask, tf.int32), 2)
            #_, cus_v = tf.dynamic_partition(cus_v, tf.cast(cus_mask, tf.int32), 2)
      
        self.debug_dict['std_mask'] = tf.shape(std_mask)
        self.debug_dict['std_v'] = [tf.shape(std_v),std_v1]
        '''
        def mask_reduce_max(tensor, mask):
            '''
            :input   tensor [batch_size, max_step, hidden_dim]
            :        seq_len [batch_size]
            :output  [batch_size, hidden_dim]
            '''
            def fn(elem):
                t, m = elem[0],elem[1]
                masked_t = tf.boolean_mask(t, m)
                o = tf.reduce_max(masked_t, axis = 0)
                return o
            output = tf.map_fn(fn, [tensor, mask])
            return output


        if std_mask is None:
            hdim = tf.shape(std_v)[-1]
            std_mask = tf.tile(tf.expand_dims(std_mask,2), (1,1,hdim))
            cus_mask = tf.tile(tf.expand_dims(cus_mask,2), (1,1,hdim))
            std_max = mask_reduce_max(std_v, std_mask)
            cus_max = mask_reduce_max(cus_v, cus_mask)
        else:
            std_max = tf.reduce_max(std_v, axis = -2)
            cus_max = tf.reduce_max(cus_v, axis = -2)
        std_avg = tf.reduce_sum(std_v, axis = -2)/tf.expand_dims(tf.cast(self.std_que_len,tf.float32),-1)
        cus_avg = tf.reduce_sum(cus_v, axis = -2)/tf.expand_dims(tf.cast(self.cus_que_len,tf.float32),-1)
        v = tf.concat([std_avg, std_max, cus_avg, cus_max], axis = -1)
        v = tf.reshape(v, shape = [self.batch_size, 4*2*self.lstm_dim])
        return v

    def optimizer_layer(self):
        with tf.variable_scope("optimizer"):
            optimizer = self.config.optimizer
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            self.train_op = self.opt.apply_gradients(grads_vars, self.global_step)
            #grads_vars = self.opt.compute_gradients(self.loss)
            '''
            capped_grads_vars = [[tf.clip_by_value(g, -self.clip, self.clip), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
            '''

    def train(self, sess, data, config):
        '''
        input : data: list of tensors, including std, cus, label
        '''
        
        data = zip(*data)
        data = [list(d) for d in data]
        _, loss, global_step,predictions,summary,debug_dict,acc = sess.run([self.train_op, self.loss, self.global_step, self.predictions,self.merged,self.debug_dict,self.acc], 
                            feed_dict = {self.std_que:data[0],
                            self.cus_que:data[1],
                            self.labels:data[2],
                            self.dropout_keep_prob: self.train_drop_keep_prob})
        #print('debug_dict:',debug_dict)
        return loss,acc#, global_step, predictions#,summary

    def evalue(self, sess, data,config):
        data = zip(*data)
        data = [list(d) for d in data]
        loss,predictions = sess.run([self.loss, self.predictions], 
                            feed_dict = {self.std_que:data[0],
                            self.cus_que:data[1],
                            self.labels:data[2],self.dropout_keep_prob: 1.0
                            })
        return loss,predictions
    '''
    def predict(self, sess, data):
        prediction = sess.run([self.predictions], feed_dict = {self.std_que:data[0],
                            self.cus_que:data[1],
                            self.labels:np.array([[[0]]])})
        return prediction
    '''
    def predict(self, sess, std, cus):
        inference = sess.run([self.predictions], feed_dict = {self.std_que: std,
                            self.cus_que:cus,
                            self.labels:np.array([0]),self.dropout_keep_prob:1.0})
        return inference


if __name__ == "__main__":
    pass
'''
    to be modified: mask attention
                    decoded length
                    check hidden dim
                    multi-layer perception
'''
