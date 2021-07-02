import numpy as np
import pandas as pd
import copy
class nn:
    #Scaling function
    def scale(self,X,y):
        mean=np.mean(X,axis=0)
        var=np.var(X,axis=0)
        X=X-mean
        X/=(var)**0.5
        return X.T,y.T,mean,var
    def scale_transform(self,X,mean,var):
        X=X-mean
        X/=(var)**0.5
        return X.T
    
    #Activation functions and their derivatives
    #Sigmoid
    def sigmoid(self,z):
        return 1/(1+(np.exp(-z)))
    def sigmoid_derivative(self,a):
        return self.sigmoid(a)*(1-self.sigmoid(a))
    
    #tanh
    def tanh(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    def tanh_derivative(self,a):
        return 1-self.tanh(a)**2
    
    #relu
    def relu(self,z):
        out=copy.deepcopy(z)
        out[out<0]=0
        return out
    def relu_derivative(self,a):
        out=copy.deepcopy(a)
        out[out>0]=1
        out[out<=0]=0
        return out
    
    #leaky_relu
    def leaky_relu(self,z,c=0.01):
        out=copy.deepcopy(z)
        return np.where(out>0, out, out * c)
    def leaky_relu_derivative(self,a,c=0.01):
        out=np.ones_like(a)
        out[a<0]=c
        return out
    
    #Softmax
    def softmax(self,z):
        t=np.exp(z)
        out=t/sum(t)
        return out
    #Softmax derivative(Not actually derivative but helper function to find dz)
    def softmax_derivative(self,y,a):
        return a-y
    
    #Loss function and its derivative
    def log_loss(self,y,a):
        return (-y *np.log(a+10**-8)-(1-y)*np.log(1 - a+10**-8)).mean()
    
    def log_loss_derivative(self,y,a):
        return (-y/a)+((1-y)/(1-a))
    
    #Random Weights Initialisation function
    def init_weights(self,nodes,features):
        return np.random.randn(nodes,features)
    
    
    def __init__(self,nodes, activations):
        np.random.seed(0)
        self.nodes=nodes
        self.activations=activations    
        self.act_func={'sigmoid':self.sigmoid,'tanh':self.tanh,'relu':self.relu,'leaky_relu':self.leaky_relu,'softmax':self.softmax}
        self.act_func_der={'sigmoid':self.sigmoid_derivative,'tanh':self.tanh_derivative,'relu':self.relu_derivative,
                  'leaky_relu':self.leaky_relu_derivative,'softmax':self.softmax_derivative}
        self.w=[0]
        self.b=[0]
        features=1
        for layer in range(len(self.nodes)):
            self.w.append(self.init_weights(self.nodes[layer],features))
            self.b.append(np.zeros((self.nodes[layer],1)))
            features=self.nodes[layer]
                
    def predict(self,X):
        X=X.T
        Layers_num=len(self.nodes)
        z=[0]
        a=[X]
        for layer in range(1,Layers_num+1):
            z.append(np.dot(self.w[layer],a[layer-1])+self.b[layer])
            a.append(self.act_func[self.activations[layer-1]](z[layer]))
        
        output=copy.deepcopy(a[-1])
        return output.T

import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from dataloader_iam import Batch

tf.compat.v1.disable_eager_execution()

class Model:
    """Minimalistic TF model for HTR."""

    def __init__(self,
                 char_list: List[str],
                 decoder_type: str = DecoderType.BestPath,
                 must_restore: bool = False,
                 dump: bool = False) -> None:
        """Init model: add CNN, RNN and CTC and initialize TF."""
        self.dump = dump
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_ID = 0

        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        self.sess, self.saver = self.setup_tf()

    def setup_cnn(self) -> None:
        cnn_in4d = tf.expand_dims(input=self.input_imgs, axis=3)

        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        pool = cnn_in4d 
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.random.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]],
                                           stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                                    strides=(1, stride_vals[i][0], stride_vals[i][1], 1), padding='VALID')

        self.cnn_out_4d = pool

    def setup_rnn(self) -> None:
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in
                 range(2)] 
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in3d,
                                                                dtype=rnn_in3d.dtype)
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        kernel = tf.Variable(tf.random.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_out_3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                                     axis=[2])

    def setup_ctc(self) -> None:
        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
        self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2]))

        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.ctc_in_3d_tbc,
                                                  sequence_length=self.seq_len,
                                                  ctc_merge_repeated=True))

        self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32,
                                                        shape=[None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input,
                                                         sequence_length=self.seq_len, ctc_merge_repeated=True)
        if self.decoder_type == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)
        elif self.decoder_type == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len,
                                                         beam_width=50)
        elif self.decoder_type == DecoderType.WordBeamSearch:
            chars = ''.join(self.char_list)
            word_chars = open('model/wordCharList.txt').read().splitlines()[0]
            corpus = open('data/corpus.txt').read()
            from word_beam_search import WordBeamSearch
            self.decoder = WordBeamSearch(50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                          word_chars.encode('utf8'))
            self.wbs_input = tf.nn.softmax(self.ctc_in_3d_tbc, axis=2)

    def setup_tf(self) -> Tuple[tf.compat.v1.Session, tf.compat.v1.train.Saver]:
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session() 

        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        model_dir = 'model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir) 

        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

    def to_sparse(self, texts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
        """Put ground truth texts into sparse tensor for ctc_loss."""
        indices = []
        values = []
        shape = [len(texts), 0]  

        for batchElement, text in enumerate(texts):

            label_str = [self.char_list.index(c) for c in text]

            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            for i, label in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoder_output_to_text(self, ctc_output: tuple, batch_size: int) -> List[str]:

        if self.decoder_type == DecoderType.WordBeamSearch:
            label_strs = ctc_output

        else:
            decoded = ctc_output[0][0]
            label_strs = [[] for _ in range(batch_size)]

            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batch_element = idx2d[0] 
                label_strs[batch_element].append(label)

        return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]

    def train_batch(self, batch: Batch) -> float:
        """Feed a batch into the NN to train it."""
        num_batch_elements = len(batch.imgs)
        max_text_len = batch.imgs[0].shape[0] // 4
        sparse = self.to_sparse(batch.gt_texts)
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_imgs: batch.imgs, self.gt_texts: sparse,
                     self.seq_len: [max_text_len] * num_batch_elements, self.is_train: True}
        _, loss_val = self.sess.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_val

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        dump_dir = 'dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                for c in range(max_c):
                    csv += str(rnn_output[t, b, c]) + ';'
                csv += '\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
        num_batch_elements = len(batch.imgs)

        eval_list = []

        if self.decoder_type == DecoderType.WordBeamSearch:
            eval_list.append(self.wbs_input)
        else:
            eval_list.append(self.decoder)

        if self.dump or calc_probability:
            eval_list.append(self.ctc_in_3d_tbc)

        max_text_len = batch.imgs[0].shape[0] // 4
        feed_dict = {self.input_imgs: batch.imgs, self.seq_len: [max_text_len] * num_batch_elements,
                     self.is_train: False}


        eval_res = self.sess.run(eval_list, feed_dict)

        if self.decoder_type != DecoderType.WordBeamSearch:
            decoded = eval_res[0]

        else:
            decoded = self.decoder.compute(eval_res[0])

        texts = self.decoder_output_to_text(decoded, num_batch_elements)
        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gt_texts) if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
                         self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}
            loss_vals = self.sess.run(eval_list, feed_dict)
            probs = np.exp(-loss_vals)

        if self.dump:
            self.dump_nn_output(eval_res[1])

        return texts, probs

    def save(self) -> None:
        self.snap_ID += 1
        self.saver.save(self.sess, 'model/snapshot', global_step=self.snap_ID)

class DecoderType:
    """CTC decoder types."""
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2
