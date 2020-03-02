# coding:utf-8
import tensorflow as tf

class bi_lstm():
    def __init__(self,hidden_size,batch_size,input):
        self.input = input
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        self.bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        self.init_fw = self.fw_cell.zero_state(self.batch_size,dtype=tf.float32)
        self.init_bw = self.bw_cell.zero_state(self.batch_size,dtype=tf.float32)
        self.outputs,self.final_states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,
                                                                         self.bw_cell,
                                                                         self.input,
                                                                         initial_state_fw = self.init_fw,
                                                                         initial_state_bw = self.init_bw
        )
        self.biLSTM_outputs = tf.layers.dense(tf.concat(self.outputs,2),self.hidden_size,activation=tf.nn.tanh,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020))
        self.state_out = tf.layers.dense(self.final_states[-1][-1],self.hidden_size,activation=tf.nn.tanh,kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020))