# coding:utf-8
import tensorflow as tf
import collections
import os
import numpy as np
import math
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tflearn.layers.conv import global_avg_pool
# Paras for BN
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'

class RECnn(object):
    def __init__(
        self,sequence_length, num_classes,
            embedding_size, filter_sizes, num_filters,embedded_chars_expanded,keep_prob,is_training=False):
        #embedded_chars_expanded是扩张维度后的输入[32,128,768,1]
        with tf.device('/gpu:0'):               
            self.keep_prob = keep_prob
            self.is_training=is_training
            self.embedded_chars_expanded = embedded_chars_expanded#加上一个维度，类似一个图片。
            #因为要卷积所以加一个维度。(32,128,786,1)
            pooled_outputs = []
            j_index=0
            for i, filter_size in enumerate(filter_sizes):#遍历卷积核[3,4,5]
                with tf.name_scope("conv-maxpool-%s" % filter_size):                     
                    filter_shape = [filter_size, embedding_size, 1, num_filters]#[3,768,1,128]卷积核形状[宽，高，输入通道，输出通道]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded, #(32,128,768,1) # NHWC
                        W,  #[3,768,1,128]# filter_height, filter_width, in_channels, out_channels
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")#输出32*126*1*128
                    #卷积之后：BN
                    c_ = {'use_bias': False,#这里我手动改成了False,即不使用bias，使用BN训练时true，
                          'is_training':self.is_training}
                          #训练时true，
                    conv = self.bn(conv,c_,'{}-bn'.format(i))#[32,126,1,128]
                    ###这里没有drop_out!
                    beta2 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='first-swish')
                    x2 = tf.nn.bias_add(conv, b)#这里的b是一个常量。
                    h = x2 * tf.nn.sigmoid(x2 * beta2)#gelu函数的变形gelu=x*sigmoid(1.702*x),只不过1.702没有写死
                    #输入[32,126,1,128]
                    for j in range(6):#6层CNNblock，每层是两个padding=same的卷积接一个SE通道加权
                        j_index +=1
                        h2 = self.Cnnblock(num_filters, h, j_index)#                          
                        h = h2+h
                        #输出不变[32,126,1,128]
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],#sequence_length - filter_size + 1=宽
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")#[32,1,1,128]
                    pooled_avg = tf.nn.avg_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")#[32,1,1,128] 
                    ###[3,4,5]三次，所以num_filters*6而不是num_filters*2
                    pooled_outputs.append(pooled)
                    pooled_outputs.append(pooled_avg)
                    
            num_filters_total = num_filters*6#128*6=768
            #在尺寸变[3,4,5]后这个地方要num_filters*2变成num_filters*6。
            self.num_filters_total=num_filters_total

            self.h_pool = tf.concat(pooled_outputs, 3)#拼接
            #[32, 1, 1, 768]
            #reshape后[32,768]
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="hidden_feature")

            #两层全连接网络
            with tf.name_scope("MLP"):
                W0 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W0")#[256,256]
                b0 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b0")
                h0 = tf.nn.relu(tf.nn.xw_plus_b(self.h_pool_flat, W0, b0))
                W1 = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="b1")
                self.h1 = tf.nn.relu(tf.nn.xw_plus_b(h0, W1, b1))#[32,768]

            with tf.name_scope("dropout"):
                self.h1 = tf.nn.dropout(self.h1,self.keep_prob)

            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],#[768,20]标签数目
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                self.scores = tf.nn.xw_plus_b(self.h1, W, b, name="scores")

    def Cnnblock(self, num_filters, h, i, has_se=True):
        W1 = tf.get_variable(
            "W1_"+str(i),
            #[3,1,128,128]
            shape=[3, 1, num_filters, num_filters],#卷积核的初始化
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1_"+str(i))
        conv1 = tf.nn.conv2d(
            h,
            W1,
            strides=[1, 1, 1, 1],#用[3,1,128,128]卷积核卷积，不改变通道数和特征图形状padding=same
            padding="SAME")
        
        c_ = {'use_bias': True, 'is_training': self.is_training}
        conv1 = self.bn(conv1, c_, str(i) + '-conv1')#BN
        #经过第一个卷积层，形状不变[32,126,1,128]
        ###这里又没drop_out!!!
        beta1 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='swish-beta-{}-1'.format(i))                      

       
        x1 = tf.nn.bias_add(conv1, b1)
        h1 = x1 * tf.nn.sigmoid(x1 * beta1)
        #上面是self_Attention神操作,乘bate1的目的是，让特征图的值分布更加分散，进入sigmoid就能产生差距较大的权重，
        #从而与自身对位相乘，进行self_Attention
        W2 = tf.get_variable(
            "W2_"+str(i),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2_"+str(i))
        conv2 = tf.nn.conv2d(
            h1,
            W2,
            strides=[1, 1, 1, 1],
            padding="SAME")

        conv2 = self.bn(conv2, c_, str(i) + '-conv2')
        
        beta2 = tf.Variable(tf.truncated_normal([1], stddev=0.08), name='swish-beta-{}-2'.format(i))
        x2 = tf.nn.bias_add(conv2, b2)
        h2 = x2 * tf.nn.sigmoid(x2 * beta2)
        #经过第二个self_Attention操作，形状不变[32,126,1,128]
        if has_se:
            h2 = self.Squeeze_excitation_layer(h2, num_filters, 16, 'se-block-' + str(i))
        #得到经过通道加权后的[32，126，1，128]
        return h2

    def bn(self, x, c, name):#对[32,126,1,128]的特征图进行BN
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]#128

        if c['use_bias']:
            bias = self._get_variable('bn_bias_{}'.format(name), params_shape,
                                      initializer=tf.zeros_initializer)
            return x + bias

        axis = list(range(len(x_shape) - 1))#意思就是取除了最后一个轴的所有维度【0，1，2】因为默认是[batch,宽,高,通道数]
        beta = self._get_variable('bn_beta_{}'.format(name),
                                  params_shape,#形状[128]即通道数
                                  initializer=tf.zeros_initializer)
        gamma = self._get_variable('bn_gamma_{}'.format(name),
                                   params_shape,#形状[128]即通道数
                                   initializer=tf.ones_initializer)
        moving_mean = self._get_variable('bn_moving_mean_{}'.format(name), params_shape,
                                         initializer=tf.zeros_initializer,
                                         trainable=False)#形状[128]即通道数
        moving_variance = self._get_variable('bn_moving_variance_{}'.format(name),
                                             params_shape,#形状[128]即通道数
                                             initializer=tf.ones_initializer,
                                             trainable=False)
        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)#所谓BN,其实是对一个批次中每个实例的同一个通道的所有数即H*W*C个数求均值，得到1个数；
        #那么一共C个通道就会得到C个数，被记录下来tf.nn.moments中的axes=[]列表中需要填入除了通道数这个维度的其他所有维度
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,# variable * decay + value * (1 - decay)
                                                                   mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        
        condition = tf.convert_to_tensor(c['is_training'], dtype='bool')        
        mean, variance = control_flow_ops.cond(#condition是布尔值"True"或者"False"的Tensor格式，输入到control_flow_ops.cond
            condition, lambda: (mean, variance),#中进行控制程序执行流，condition 是True就返回第一个，False就返回第二个
            lambda: (moving_mean, moving_variance))#意思就是训练=True,返回当前批次均值方差mean,variance;不是训练过程
                                                   #就返回历史记录均值方差moving_mean/variance
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

        return x

    def _get_variable(self, name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):#SE层
        
        with tf.name_scope(layer_name):
            squeeze = self.Global_Average_Pooling(input_x)
            #平均池化[32,128]
            
            excitation = self.Fully_connected(squeeze, units=out_dim / ratio,#128/16=8
                                              layer_name=layer_name + '_fully_connected1')
            excitation = self.Relu(excitation)
            #[?,8]变成[?,128]  
            excitation = self.Fully_connected(excitation, units=out_dim,#out_dim=128
                                              layer_name=layer_name + '_fully_connected2')           
            excitation = self.Sigmoid(excitation)
            #得到32*128个sigmoid值，相当于每个通道（128）的权重
            ###这里是不是可以drop_out?
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            #[32,1,1,128]
            scale = input_x * excitation#[32,126,1,128]*[32,1,1,128]#相当于在每个通道上加权重。

            return scale

    def Global_Average_Pooling(self, x):
        return global_avg_pool(x, name='Global_avg_pooling')

    def Relu(self, x):
        return tf.nn.relu(x)

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def Fully_connected(self, x, units, layer_name='fully_connected'):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=True, units=units)#units=8=128/16
            #这里就是WX+b,输出是8维的，
            #由[?,128]变成[?,8]  



class seq_attention():
    
    def __init__(self,bert_sequence_out,keep_prob,num_classes,max_seq_lenth):
        self.bert_sequence_out = bert_sequence_out
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.max_seq_length = max_seq_lenth
        
        
                
        with tf.name_scope("seq_attention"):
            hidden_size = self.bert_sequence_out.shape[2].value  # D value - hidden size of the RNN layer
            value = tf.layers.dense(self.bert_sequence_out,hidden_size)
            quarry = tf.layers.dense(self.bert_sequence_out,hidden_size)
            key = tf.layers.dense(self.bert_sequence_out,hidden_size)
            value = tf.transpose(tf.reshape(value,[-1,self.max_seq_length,16,64]),[0,2,1,3])
            quarry = tf.transpose(tf.reshape(quarry,[-1,self.max_seq_length,16,64]),[0,2,1,3])
            key = tf.transpose(tf.reshape(key,[-1,self.max_seq_length,16,64]),[0,2,1,3])
            q_k = tf.nn.softmax(tf.matmul(value, key,transpose_b=True)/tf.cast(8,tf.float32),-1)
            
            q_k_v = tf.matmul(value,q_k,transpose_a=True)
            q_k_v = tf.reshape(tf.transpose(q_k_v,[0,3,1,2]),[-1,self.max_seq_length,hidden_size])
            w_2 = tf.layers.dense(tf.layers.dense(q_k_v,100,activation=tf.tanh),1)
            w_2 = tf.expand_dims(tf.nn.softmax(tf.reshape(w_2,[-1,self.max_seq_length]),-1),-1)
            
            v_2 = tf.layers.dense(q_k_v,hidden_size,activation=tf.tanh)
            
            wv = tf.reshape(tf.reduce_mean(v_2*w_2,1),[-1,hidden_size])
            
            wv_ = tf.nn.dropout(tf.layers.dense(wv,hidden_size),self.keep_prob)
            logits = tf.layers.dense(wv_,self.num_classes)
            self.logits = logits

    
def old_attention(bert_output,bert_hx,attention_size,max_seq_length,keep_prob):
     
    hidden_size = bert_output.shape[2].value  # D value - hidden size of the RNN layer
    

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size*2, attention_size], stddev=0.1))  # W=[768*2,50]
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  # b=[50]
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  # U=[50,]

    batch_=tf.concat([bert_output,tf.tile(tf.expand_dims(bert_hx,1),[1,max_seq_length,1])],2)
    #tf.expand_dims(hx,1)在[32,768]中间加一个维度，变成[32,1,768]
    #tf.tile(tf.expand_dims(hx,1),[1,self.config.seq_length,1])-复制最后一个时刻的输出[32,1,768]变成[32,128,768]
    #batch_=[32,128,1536]加上了第一个字的效果。

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    v = tf.tanh(tf.tensordot(batch_, w_omega, axes=1) + b_omega)#压缩特征到50
    #tf.einsum('ijm,mn->ijn', inputs, w_omega)
        
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    # 
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # 继续压缩特征，讲50维度的特征全部压缩到1
    # [32，128]
    alphas = tf.nn.softmax(vu, name='alphas')  # softmax;总权重和为1      

   
    bert_attention_output = tf.reduce_sum( bert_output* tf.expand_dims(alphas, -1), 1)#[32,128,768]*[32,128,1]把每个字的权重
    #对位相乘到bert_out128个字的表征中，再把句话128个字的特征求和成1个字(标签)的特征，即[32,768]


    fc = tf.layers.dense(bert_attention_output, hidden_size, name='fc1')#全连接[768,768]
    fc = tf.nn.dropout(fc,keep_prob=keep_prob)
    fc = tf.nn.relu(fc)
    #输出[32,768]
    # [?, 20] 分类置信度
    logits_rnn = tf.layers.dense(fc, 3, name='fc2')
    return logits_rnn

def attention_layer(input_tensor,input_mask,attention_head,size_per_head,attention_keep_probs):
    tensor_shape = tf.shape(input_tensor)
    batch_size = tensor_shape[0]
    seq_len = tensor_shape[1]
    hidden_size = tensor_shape[2]
    # quarry
    quarry = tf.transpose(tf.reshape(tf.layers.dense(input_tensor,attention_head*size_per_head),[batch_size,seq_len,attention_head,size_per_head]),[0,2,1,3])
    # key
    key = tf.transpose(tf.reshape(tf.layers.dense(input_tensor,attention_head*size_per_head),[batch_size,seq_len,attention_head,size_per_head]),[0,2,1,3])
    # value
    value = tf.transpose(tf.reshape(tf.layers.dense(input_tensor,attention_head*size_per_head),[batch_size,seq_len,attention_head,size_per_head]),[0,2,1,3])
    # scores
    scores = tf.matmul(quarry,key,transpose_b=True)
    scores = tf.multiply(scores,1.0 / math.sqrt(float(size_per_head)))    
    # mask [bs,1,1,seq]
    mask = tf.expand_dims(tf.expand_dims(input_mask,axis=[1]),axis=[1])
    adder = (1.0 - tf.cast(mask, tf.float32)) * -10000.0
    # attention_probs
    scores += adder
    attention_probs = tf.nn.dropout(tf.nn.softmax(scores),attention_keep_probs)
    # out
    attention_out = tf.matmul(attention_probs,value)
    attention_out = tf.reshape(tf.transpose(attention_out,[0,2,1,3]),[batch_size,seq_len,hidden_size])
    #
    return attention_out

def cnn_layers(input,kernel_size,is_train,name,second_pad_mode):
    #if kernel_size == 1:
        #stride = 1
    #elif kernel_size == 3:
        #stride = 2
    #else:
        #stride = 3
    # 第一个卷积层
    with tf.variable_scope(name+'conv_1'):        
        w1 = tf.get_variable('w1',[kernel_size,kernel_size,1,64],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer())
    conv_1 = tf.nn.conv2d(input,w1,strides=[1,1,1,1],padding='SAME')
    conv_1 = tf.nn.bias_add(conv_1,b1)
    conv_1 = tf.layers.batch_normalization(conv_1,training=is_train)
    conv_1 = tf.nn.sigmoid(1.702*conv_1)*conv_1
    # 第一个池化层
    #high = conv_1.shape[1]
    conv_1_maxpool = tf.nn.max_pool(conv_1, [1,3,3,1], [1,2,2,1],padding='SAME')
    #conv_1_maxpool = tf.reduce_max(conv_1,axis=1,keepdims=True)
    # 第二个卷积层
    
    with tf.variable_scope(name+'conv_2'):
        w2 = tf.get_variable('w2',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2',[128],initializer=tf.constant_initializer())
    conv_2 = tf.nn.conv2d(conv_1_maxpool,w2,strides=[1,2,2,1],padding=second_pad_mode)
    conv_2 = tf.nn.bias_add(conv_2,b2)
    conv_2 = tf.layers.batch_normalization(conv_2,training=is_train)
    conv_2 = tf.nn.sigmoid(1.702*conv_2)*conv_2
    # 第二个池化层
    #wigth = conv_2.shape[2]
    conv_2_max_pool = tf.nn.max_pool(conv_2,[1,3,3,1],[1,2,2,1],padding='SAME')
    reduced_max_ = tf.reshape(tf.reduce_max(tf.reduce_max(conv_2_max_pool,axis=1,keepdims=True),axis=2,keepdims=True),[-1,128])
    reduced_mean_ = tf.reshape(tf.reduce_mean(tf.reduce_mean(conv_2_max_pool,axis=1,keepdims=True),axis=2,keepdims=True),[-1,128])
    #conv_2_max_pool = tf.reduce_max(conv_2,axis=2,keepdims=False)
    #out = tf.reshape(conv_2_max_pool,[-1,256])
    return reduced_max_,reduced_mean_

def comprehention_cnn(is_train,bert_pooled_out,bert_seq_out,input_mask,attention_head,size_per_head,attention_keep_probs):
    text_matrix = bert_seq_out[:,47:,:]  # text的向量
    topic_matrix = bert_seq_out[:,1:46,:] # 除去cls和sep
    attention_topic = attention_layer(topic_matrix, input_mask[:,1:46], attention_head, size_per_head, attention_keep_probs)
    attention_text = attention_layer(text_matrix, input_mask[:,47:], attention_head, size_per_head, attention_keep_probs)
    cnn_matrix = tf.matmul(attention_text,attention_topic,transpose_b=True) # [bs,45,133]
    # 卷积部分
    cnn_input = tf.expand_dims(cnn_matrix,axis=[-1])
    cnn_kernel_1_max,cnn_kernel_1_mean = cnn_layers(cnn_input, 1, is_train, 'cnn_kernel_1','VALID')
    cnn_kernel_2_max,cnn_kernel_2_mean= cnn_layers(cnn_input, 2, is_train, 'cnn_kernel_3','VALID')
    cnn_kernel_3_max,cnn_kernel_3_mean = cnn_layers(cnn_input, 3, is_train, 'cnn_kernel_5','SAME')
    # 与CLS向量进行CONCAT
    concat_matrix = tf.concat([bert_pooled_out,cnn_kernel_1_max,cnn_kernel_1_mean,cnn_kernel_2_max,cnn_kernel_2_mean,cnn_kernel_3_max,cnn_kernel_3_mean],axis=-1)
    # 前馈网路
    intermidie_layers = tf.layers.dense(concat_matrix,100,activation=tf.nn.tanh)
    logits = tf.layers.dense(intermidie_layers,3)
    return logits
    
    

