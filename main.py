# -*- coding: utf-8 -*

import argparse
import tensorflow as tf
from flyai.dataset import Dataset
# from flyai.utils import remote_helper
import load_bert_model
from model import Model
from path import MODEL_PATH, LOG_PATH
import bert.modeling as modeling
import os
import sys
# from data_augment import convertText
from sklearn import metrics
import random
from flyai.utils.log_helper import train_log
import data_augment
from restnet_cnn_attention import RECnn
from restnet_cnn_attention import seq_attention
from restnet_cnn_attention import old_attention
from restnet_cnn_attention import comprehention_cnn
from lstm_cnn_net import bi_lstm
import numpy as np
import re
import jieba
import gensim
import json
# ---------超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()
# ---------数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# ---------模型操作辅助类
modelpp = Model(dataset)
# 统计数据的分布

train_x,train_label,val_x,val_label = dataset.get_all_data()

topic_a = {'topic':'深圳禁摩限电','total_number':0,'None':0,'Favor':0,'Agan':0,'text':[]}
topic_b = {'topic':'春节放鞭炮','total_number':0,'None':0,'Favor':0,'Agan':0,'text':[]}
topic_c = {'topic':'IphoneSE','total_number':0,'None':0,'Favor':0,'Agan':0,'text':[]}
topic_d = {'topic':'开放二胎','total_number':0,'None':0,'Favor':0,'Agan':0,'text':[]}
topic_e = {'topic':'俄罗斯在叙利亚的反恐行动','total_number':0,'None':0,'Favor':0,'Agan':0,'text':[]}
total = {'total':0,'None':0,'Favor':0,'Agan':0}
for i in range(len(train_x)):
    if train_x[i]['TARGET'] == '深圳禁摩限电':     
        if train_label[i]['STANCE'] == 'AGAINST':
            topic_a['Agan'] += 1
            #topic_a['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'FAVOR':
            topic_a['Favor'] += 1
            #topic_a['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'NONE':
            topic_a['None'] += 1
            #f_record_none.write(train_x[i]['TEXT']+'\n')
        topic_a['total_number'] += 1
        topic_a['text'].append(train_x[i]['TEXT'])
    if train_x[i]['TARGET'] == '春节放鞭炮':
        if train_label[i]['STANCE'] == 'AGAINST':
            topic_b['Agan'] += 1
            #topic_b['text'].append(train_x[i]['TEXT']) 
        if train_label[i]['STANCE'] == 'FAVOR':
            topic_b['Favor'] += 1
            #topic_b['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'NONE':
            topic_b['None'] += 1
            #f_record_none.write(train_x[i]['TEXT']+'\n')
        topic_b['total_number'] += 1
        topic_b['text'].append(train_x[i]['TEXT'])
    if train_x[i]['TARGET'] == 'IphoneSE':
        if train_label[i]['STANCE'] == 'AGAINST':
            topic_c['Agan'] += 1
            #topic_c['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'FAVOR':
            topic_c['Favor'] += 1
            #topic_c['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'NONE':
            topic_c['None'] += 1
            #f_record_none.write(train_x[i]['TEXT']+'\n')
        topic_c['total_number'] += 1
        topic_c['text'].append(train_x[i]['TEXT'])
    if train_x[i]['TARGET'] == '开放二胎':
        if train_label[i]['STANCE'] == 'AGAINST':
            topic_d['Agan'] += 1
            #topic_d['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'FAVOR':
            topic_d['Favor'] += 1
            #topic_d['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'NONE':
            topic_d['None'] += 1
            #f_record_none.write(train_x[i]['TEXT']+'\n')
        topic_d['total_number'] += 1 
        topic_d['text'].append(train_x[i]['TEXT'])
    if train_x[i]['TARGET'] == '俄罗斯在叙利亚的反恐行动':
        if train_label[i]['STANCE'] == 'AGAINST':
            topic_e['Agan'] += 1
            #topic_e['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'FAVOR':
            topic_e['Favor'] += 1
            #topic_e['text'].append(train_x[i]['TEXT'])
        if train_label[i]['STANCE'] == 'NONE':
            topic_e['None'] += 1
            #f_record_none.write(train_x[i]['TEXT']+'\n')
        topic_e['total_number'] += 1
        topic_e['text'].append(train_x[i]['TEXT'])
for i in range(len(val_x)):
    if val_x[i]['TARGET'] == '深圳禁摩限电':
        if val_label[i]['STANCE'] == 'AGAINST':
            topic_a['Agan'] += 1
            #topic_a['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'FAVOR':
            topic_a['Favor'] += 1
            #topic_a['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'NONE':
            topic_a['None'] += 1
            #f_record_none.write(val_x[i]['TEXT']+'\n')
        topic_a['total_number'] += 1
        topic_a['text'].append(val_x[i]['TEXT'])
    if val_x[i]['TARGET'] == '春节放鞭炮':
        if val_label[i]['STANCE'] == 'AGAINST':
            topic_b['Agan'] += 1
            #topic_b['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'FAVOR':
            topic_b['Favor'] += 1
            #topic_b['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'NONE':
            topic_b['None'] += 1
            #f_record_none.write(val_x[i]['TEXT']+'\n')
        topic_b['total_number'] += 1  
        topic_b['text'].append(val_x[i]['TEXT'])
    if val_x[i]['TARGET'] == 'IphoneSE':
        if val_label[i]['STANCE'] == 'AGAINST':
            topic_c['Agan'] += 1
            #topic_c['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'FAVOR':
            topic_c['Favor'] += 1
            #topic_c['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'NONE':
            topic_c['None'] += 1
            #f_record_none.write(val_x[i]['TEXT']+'\n')
        topic_c['total_number'] += 1
        topic_c['text'].append(val_x[i]['TEXT'])
    if val_x[i]['TARGET'] == '开放二胎':
        if val_label[i]['STANCE'] == 'AGAINST':
            topic_d['Agan'] += 1
            #topic_d['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'FAVOR':
            topic_d['Favor'] += 1
            #topic_d['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'NONE':
            topic_d['None'] += 1
            #f_record_none.write(val_x[i]['TEXT']+'\n')
        topic_d['total_number'] += 1 
        topic_d['text'].append(val_x[i]['TEXT'])
    if val_x[i]['TARGET'] == '俄罗斯在叙利亚的反恐行动':
        if val_label[i]['STANCE'] == 'AGAINST':
            topic_e['Agan'] += 1
            #topic_e['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'FAVOR':
            topic_e['Favor'] += 1
            topic_e['text'].append(val_x[i]['TEXT'])
        if val_label[i]['STANCE'] == 'NONE':
            topic_e['None'] += 1
            #f_record_none.write(val_x[i]['TEXT']+'\n')
        topic_e['total_number'] += 1
        topic_e['text'].append(val_x[i]['TEXT'])

total['None'] = topic_a['None']+topic_b['None']+topic_c['None']+topic_d['None']+topic_e['None']
total['Favor'] = topic_a['Favor']+topic_b['Favor']+topic_c['Favor']+topic_d['Favor']+topic_e['Favor']
total['Agan'] = topic_a['Agan']+topic_b['Agan']+topic_c['Agan']+topic_d['Agan']+topic_e['Agan']
total['total'] = total['None']+total['Favor']+total['Agan']

print(total)

# 使用LDA抽取每个话提的主题关键词,作为BERT的text_a输入到模型中,建立text_b是否是围绕text_a展开的0，1分类任务
def clean(text):
    # return a list of jieba cutted words
    text = re.sub("[^\u4e00-\u9fa5]","",text).lower()
    f = open('stop_words.txt','r',encoding='utf-8')
    stop_words_list = []
    while True:
        line = f.readline().strip()
        if not line:
            break
        stop_words_list.append(line)
    cutted = list(jieba.cut(text))
    cutted_new = []
    for i in range(len(cutted)):
        if cutted[i] in stop_words_list or len(cutted[i])<2:
            continue
        cutted_new.append(cutted[i])
    f.close()
    return cutted_new
def get_lda_topics(doc):
    cleaned_docs = list(map(clean,doc))
    new = []
    for li in cleaned_docs:
        new += li
    dic = gensim.corpora.Dictionary([new])
    bow_corpus = [dic.doc2bow(doc) for doc in [new]]
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                           num_topics=2,
                                           id2word=dic)
    return lda_model
topic_a_model = get_lda_topics(topic_a['text'])
topic_b_model = get_lda_topics(topic_b['text'])
topic_c_model = get_lda_topics(topic_c['text'])
topic_d_model = get_lda_topics(topic_d['text'])
topic_e_model = get_lda_topics(topic_e['text'])

topic_dic = {}
topic_dic['深圳禁摩限电'] = re.sub('[^A-za-z\u4e00-\u9fa5]','',topic_a_model.print_topic(0,topn=15))
topic_dic['春节放鞭炮'] = re.sub('[^A-za-z\u4e00-\u9fa5]','',topic_b_model.print_topic(0,topn=15))
topic_dic['IphoneSE'] = re.sub('[^A-za-z\u4e00-\u9fa5]','',topic_c_model.print_topic(0,topn=15))
topic_dic['开放二胎'] = re.sub('[^A-za-z\u4e00-\u9fa5]','',topic_d_model.print_topic(0,topn=15))
topic_dic['俄罗斯在叙利亚的反恐行动'] = re.sub('[^A-za-z\u4e00-\u9fa5]','',topic_e_model.print_topic(0,topn=15))
with open('topic.txt','w',encoding='utf-8') as f:
    json.dump(topic_dic,f)
print(topic_dic['深圳禁摩限电'])
print(topic_dic['春节放鞭炮'])
print(topic_dic['IphoneSE'])
print(topic_dic['开放二胎'])
print(topic_dic['俄罗斯在叙利亚的反恐行动'])

data_root = load_bert_model.get_remote_date('https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip')
# 最新的全词遮罩bert,roberta_wwm_large_ext最优64的评分，最后一层神经元的数量是1024
#data_root = load_bert_model.get_remote_date('https://www.fly ai.com/m/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip')
#data_root = './chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'
#data_root = "chinese_L-12_H-768_A-12"
''' 
使用tensorflow实现自己的算法
'''
# 参数
# learning_rate = 0.0006
learning_rate_ = 1e-5 # 学习率0.00001
#learning_rate = 0.1 # 0.0005
num_labels = 3  # 类别数
num_labels_cls = 2
max_seq_length = 180
# ——————————————————配置文件——————————————————
# 分离文件名与扩展名
# 得到预训练模型的参数
# os.path.splitext(path)[0]
# data_root = path
bert_config_file = os.path.join(data_root, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
bert_vocab_file = os.path.join(data_root, 'vocab.txt')

with open("pre_trained_root.txt","w") as f:
    f.write(bert_vocab_file)
    f.close()  

# ——————————————————导入数据——————————————————————
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
labels = tf.placeholder(tf.int32, shape=[None,2], name='labels')
is_train = tf.placeholder(tf.bool, name='is_train')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
#attention_keep_probs = tf.placeholder(tf.float32,name='attention_keep_prob')
# ——————————————————定义神经网络变量——————————————————
# 初始化BERT
model = modeling.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

# 加载bert模型
tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)
# 获取最后一层。
# 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner用这个
# output_layer_seq = model.get_sequence_output()  # 这个获取每个token的output
# 这个获取句子的output
output_layer = tf.identity(model.get_pooled_output(), name='output_layer_pooled')
output_seq_layer = tf.identity(model.get_sequence_output(),name='output_layer_seq')
#bert_out_expanded = tf.expand_dims(output_seq_layer,-1)
# 根据输出的句向量计算维度
hidden_size = output_layer.shape[-1].value
#print('最后一层神经元数量：{}'.format(hidden_size))
#==============================================下游任务模型搭建=========================
#SE_Res_textCnn = RECnn(     #卷积部分
        #sequence_length=144,#128
        #num_classes=3,
        #embedding_size=hidden_size,#768
        #filter_sizes=[3,4,5],
        #num_filters=128,#卷积输出通道
        #embedded_chars_expanded=bert_out_expanded,
        #keep_prob=keep_prob,
        #is_training=is_train)
#seq_attention_ = seq_attention(output_seq_layer,keep_prob=keep_prob,num_classes=3,max_seq_lenth=max_seq_length)
#seq_attention_logit = old_attention(output_seq_layer, output_layer, 100, 180, keep_prob)
attention_cnn_logits = comprehention_cnn(is_train, output_layer, output_seq_layer, input_mask,12,64,keep_prob)
#cls_ = tf.layers.dense(output_layer,hidden_size//2,activation=tf.nn.relu)
cls_logits = tf.layers.dense(output_layer,2)
#两个logits相加
with tf.name_scope("combine_logits"):
    #text_cnn_logit = SE_Res_textCnn.scores
    #combine_logits = text_cnn_logit + seq_attention_logit
    combine_logits  = attention_cnn_logits
#bilstm = bi_lstm(3072, args.BATCH, output_seq_layer)
#bilstm_out = bilstm.state_out
#======================================================================================

# 构建W 和 b
#output_weights = tf.get_variable(
    #"output_weights", [3072, num_labels],
    #initializer=tf.truncated_normal_initializer(stddev=0.02))

#output_bias = tf.get_variable(
    #"output_bias", [num_labels], initializer=tf.zeros_initializer())

with tf.variable_scope("predict"):
    # I.e., 0.1 dropout   0.98 rate = 1-keep_prob,这个参数设置错误，导致我调参两天无结果。！！！！！
    #output_layer = tf.layers.dropout(bilstm_out, rate=0.05, training=is_train)
    # I.e., 0.1 dropout   0.98
    #    output_layer = tf.nn.dropout(output_layer, keep_prob=0.95)

    #logits = tf.nn.bias_add(tf.matmul(output_layer, output_weights), output_bias)
    # 效果不是很理想，还是使用log_soft吧
    probs_cls = tf.nn.softmax(cls_logits)
    #pred_cls = tf.argmax(probs_cls,1,name='pred_cls')
    probs = tf.nn.softmax(combine_logits)
    #    log_probs = tf.nn.log_softmax(logits, axis=-1)
    pred = tf.argmax(probs, 1, name='pred')

with tf.name_scope("accuracy"):
    # 准确率,tf.cast() 数据转换
    correct_pred = tf.equal(labels[:,0], tf.cast(pred, tf.int32))
    not_correct_pred = tf.not_equal(labels[:,0],tf.cast(pred,tf.int32),name='not_correct_index')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

with tf.name_scope("optimize"):
    # 将label进行onehot转化
    one_hot_labels = tf.one_hot(labels[:,0], depth=num_labels, dtype=tf.float32)
    one_hot_labels_cls = tf.one_hot(labels[:,1],depth=num_labels_cls,dtype=tf.float32)
    # 构建损失函数
    per_example_loss_cls = -tf.reduce_sum(one_hot_labels_cls * tf.log(probs_cls), axis=-1)
    per_example_loss = -tf.reduce_sum(one_hot_labels * tf.log(probs), axis=-1)
    loss = tf.reduce_mean(per_example_loss) + tf.reduce_mean(per_example_loss_cls)

    # 优化器
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.noisy_linear_cosine_decay(learning_rate=learning_rate_, global_step=global_step, decay_steps=40, initial_variance=0.01, variance_decay=0.1, num_periods=2, alpha=0.0, beta=1e-7, name=None)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(gard,-2.,2.),var) for gard, var in gvs if gard is not None]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):       
        train_op = optimizer.apply_gradients(capped_gvs)

with tf.name_scope("summary"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("acc", accuracy)
    merged_summary = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    print('the total length of train dataset', dataset.get_train_length())
    print('the total length of validation dataset', dataset.get_validation_length())
    print('dataset.get_step:', dataset.get_step())

    all_train_steps = int(dataset.get_train_length() / args.BATCH) * args.EPOCHS

    current_step = 0
    acc_flag = 0
    last_provement = 0
    #    早停步骤
    eraly_stop = 100

    #    for step in range(args.EPOCHS):
    #        for batch_train in data_augment.get_batch_dataset(all_train_x,all_train_y,args.BATCH,current_step):
    for step in range(1,dataset.get_step()):
        x_train, y_train, x_val, y_val = dataset.next_batch(args.BATCH,
                                                            dataset.get_validation_length())
        val_len = x_val[0].shape[0]
        num = np.arange(x_val[0].shape[0])
        np.random.shuffle(num)
        num =list(num[:args.BATCH])
        x_input_ids__ = x_val[0][num]
        x_input_mask__ = x_val[1][num]
        x_segment_ids__ = x_val[2][num]
        x_text__ = x_val[3][num]
        y__ = y_val[num]

        x_input_ids = x_train[0]
        x_input_mask = x_train[1]
        x_segment_ids = x_train[2]
        x_text = x_train[3]

        x_input_ids_val = x_val[0]
        x_input_mask_val = x_val[1]
        x_segment_ids_val = x_val[2]
        x_text_val = x_val[3]

        # -------train model
        fetches = [train_op, loss, accuracy,pred,not_correct_pred]
        feed_dict = {input_ids: x_input_ids, input_mask: x_input_mask, segment_ids: x_segment_ids,
                     labels: y_train, keep_prob:0.5,is_train:True}
        _, loss_,acc_, pred_,wrong= sess.run(fetches, feed_dict=feed_dict)
        train_f1 = metrics.f1_score(y_train[:,0],pred_,average='weighted')
        #if step%30 == 0:
            #print(x_text[wrong],y_train[wrong])
        # -------validation model
        feed_dict_ = {input_ids: x_input_ids__, input_mask: x_input_mask__,
                      segment_ids: x_segment_ids__, labels: y__, is_train:True,keep_prob:0.5}
        _,wrong__ = sess.run([train_op,not_correct_pred], feed_dict=feed_dict_)
        #if step%30 == 0:
            #print(x_text__[wrong__],y__[wrong__])
        feed_dict_val = {input_ids: x_input_ids_val, input_mask: x_input_mask_val,
                         segment_ids: x_segment_ids_val, labels: y_val,keep_prob:1.0,is_train:False}
        valLoss, valAcc_, y_pre,wrong_ = sess.run([loss, accuracy, pred,not_correct_pred], feed_dict=feed_dict_val)
        val_f1score = metrics.f1_score(y_val[:,0], y_pre, average='weighted')
        #if step%100 == 0:
            #print(x_text_val[wrong_],y_val[wrong_])
        # -------save and print
        summary = sess.run(merged_summary, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        print(' ')
        # cur_step = str(step + 1) + "/" + str(all_train_steps)
        print('steps: {0}'.format(str(current_step) + '/' + str(all_train_steps)))
        f1_mean = (val_f1score+train_f1)/2
        train_log(train_loss=loss_, train_acc=acc_, val_loss=valLoss, val_acc=f1_mean)
        print("val_f1:{}".format(f1_mean))
        current_step += 1
        #if current_step % 100 == 0:
            #modelpp.save_model(sess, MODEL_PATH, overwrite=True) 

        #if current_step % 10 == 0:
            # 每 5 step验证一次
        if acc_flag < f1_mean:
            acc_flag = f1_mean
            modelpp.save_model(sess, MODEL_PATH, overwrite=True)
            last_provement = current_step
            print('the save model steps is : {0}'.format(
                str(current_step) + '/' + str(all_train_steps)))
            print('the model f1score is {0}'.format(f1_mean))

        if (current_step - last_provement) >= eraly_stop and f1_mean >= 0.99:
            #                早停法
            print('model early stop in steps :{0}'.format(current_step))
            print('the save model is ini step:{0}'.format(last_provement))
            break
#        else:
#            continue
#        break

