# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:16:45 2019

@author: dabing
"""
from processor import Processor
import numpy as np
import random
import jieba 
import os
import sys
#import synonyms

def data_sugment(target,text,stance,num):
    positive_words = load_words(os.path.join(sys.path[0],'positive.txt'))
    negitive_words = load_words(os.path.join(sys.path[0],'negative.txt'))
    a = ['支持','不错','满意','鼓励','赞同','喜庆','喜欢','赞成','同意','拥护','认同','还行']
    b = ['不好','失望','讨厌','反对','垃圾','不适合','嫌弃','辣鸡','丑','傻叉','悲剧','绝望','禁止','肮脏','强奸','绝情','推翻','!']
#    for index, dic in enumerate(train_y):
#        
#        if dic['STANCE'] == 'FAVOR':
##            train_x[index]['TARGET'] = random.choice(a)+train_x[index]['TARGET']
#            train_x[index]['TEXT'] = add_positive(train_x[index]['TEXT'],train_x[index]['TARGET'],num,a,positive_words)
#        elif dic['STANCE'] == 'AGAINST':
##            train_x[index]['TARGET'] = random.choice(b)+train_x[index]['TARGET']
#            train_x[index]['TEXT'] = add_negitive(train_x[index]['TEXT'],train_x[index]['TARGET'],num,b,negitive_words)
    if stance == 'FAVOR':
        text = add_positive(text,target,num,a,positive_words)
    elif stance == 'AGAINST':
        text = add_positive(text,target,num,b,negitive_words)
        
    return target, text

#def add_positive(text,target,num,a,positive_words):
    #cut_segment = jieba.lcut(text)
    #add_words = []
##    加入1个立场词
    #random_words = random.choices(a,k=num)
    #add_words.extend(random_words)
##   至多加入2个文本中的情感词
    #for word in cut_segment:
        #if word in positive_words:
            #synonym_list = synonyms.nearby(word)[0]
            #if len(synonym_list) >= 1:
                #add_words.append(random.choice(synonym_list))
        #if len(add_words) >= num+2 :
            #break

    #for index, word in enumerate(add_words):
        ##随机index
        #random_index = random.randint(0,len(cut_segment))
        ##平均index       
        #insert_index = index*(int(len(cut_segment)/len(add_words)))
        
        #word = '/' + word + '/'
        #cut_segment.insert(0,word)
       
    #return ''.join(cut_segment)
    
#def add_negitive(text,target,num,b,negitive_words):
    #cut_segment = jieba.lcut(text)
    #add_words = []
##    加入1个立场词
    #random_words = random.choices(b,k=num)
    #add_words.extend(random_words)
##   至多加入2个文本中的情感词
    #for word in cut_segment:
        #if word in negitive_words:
            #synonym_list = synonyms.nearby(word)[0]
            #if len(synonym_list) >= 1:
                #add_words.append(random.choice(synonym_list))
        #if len(add_words) >= num+2 :
            #break

    #for index, word in enumerate(add_words):
        ##随机index
        #random_index = random.randint(0,len(cut_segment))
        
        #insert_index = index*(int(len(cut_segment)/len(add_words)))
        #word = '/' + word + '/'
        #cut_segment.insert(0,word)
       
    #return ''.join(cut_segment)
    
    
    
#def data_process(train_x,train_y):
#    new_train_x = []
#    new_train_y = []
#    ids = []
#    mask = []
#    seg_ids = []
#    train_process = Processor()
#    
#    for index, record in enumerate(train_x):
#        train_ids, train_mask, segment_ids =  train_process.input_x(record['TARGET'],record['TEXT'])
#        index_y = train_process.input_y(train_y[index]['STANCE'])
#        ids.append(train_ids)
#        mask.append(train_mask)
#        seg_ids.append(segment_ids)
#        new_train_y.append(index_y)    
#    
#    new_train_x.append(np.array(ids))
#    new_train_x.append(np.array(mask))
#    new_train_x.append(np.array(seg_ids))
#    new_train_y = np.array(new_train_y)
#    
#    return new_train_x, new_train_y
#
#def get_batch_dataset(train_x,train_y,batch_size,flag):
#    '''
#    返回的是迭代对象，返回两个则放在一个列表中，使用[0][1]提取
#    '''
#    batch_num = int(len(train_y)/batch_size)
#    
#    for index in range(batch_num):
##        原始batch 数据
#        batch_x = train_x[index*batch_size:(index+1)*batch_size]
#        batch_y = train_y[index*batch_size:(index+1)*batch_size]
#        
##        大于60， 增加观点词
#        if flag >= 60:
#            batch_x, batch_y = data_sugment(batch_x,batch_y,num=1)
##       获取编码形式
#        mini_batch_x, mini_batch_y = data_process(batch_x,batch_y)
#        
##返回迭代对象        
#        yield mini_batch_x, mini_batch_y



def load_words(path):
    words = []
    with open(path,'r',encoding ='utf-8') as file:
        for line in file.readlines():
            words.append(line.strip())

    return words