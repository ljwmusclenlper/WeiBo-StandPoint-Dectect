# -*- coding: utf-8 -*
import os
from path import DATA_PATH
from flyai.processor.base import Base
import bert.tokenization as tokenization
from bert.run_classifier import convert_single_example_simple
import re
import json
pattern = "[^A-za-z0-9\u4e00-\u9fa5\！\？\。\，\：\‘\’\；\“\”\、\,\.\:\;\'\"\!\?\<\>]"


class Processor(Base):
    def __init__(self):
        self.token = None             
                
    def input_x(self, TARGET, TEXT):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        #with open('topic.txt','r',encoding='utf-8') as f:
            #topic = json.load(f)
        TARGET = re.sub(pattern,"",TARGET)
        TEXT = re.sub(pattern,"",TEXT)
        #if TARGET == '深圳禁摩限电':
            #TARGET+= topic['深圳禁摩限电']
        #elif TARGET == '春节放鞭炮':
            #TARGET+= topic['春节放鞭炮']
        #elif TARGET == 'IphoneSE':
            #TARGET+= topic['IphoneSE']
        #elif TARGET == '开放二胎':
            #TARGET+= topic['开放二胎']
        #else:
            #TARGET+= topic['俄罗斯在叙利亚的反恐行动']
        with open("pre_trained_root.txt","r") as f:
            vocab_root = f.readline()
            f.close()
        if self.token is None:
            #bert_vocab_file = os.path.join(DATA_PATH, "model", "multi_cased_L-12_H-768_A-12", 'vocab.txt')
            self.token = tokenization.CharTokenizer(vocab_file=vocab_root)
        word_ids, word_mask, word_segment_ids = \
            convert_single_example_simple(max_seq_length=180, tokenizer=self.token, text_a=TARGET, text_b=TEXT)
        return word_ids, word_mask, word_segment_ids,TARGET+TEXT

    def input_y(self, STANCE):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        if STANCE == 'NONE':
            return [0,0]
        elif STANCE == 'FAVOR':
            return [1,1]
        elif STANCE == 'AGAINST':
            return [2,1 ]

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return data[0]