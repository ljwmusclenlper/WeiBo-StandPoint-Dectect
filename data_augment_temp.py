# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:35:43 2019

@author: dabing

使用翻译接口，对文本数据进行回译，用来进行训练样本的数据量扩充
"""
import hashlib
import urllib
import json
from translate import Translator

import jieba
import synonyms
import random
from random import shuffle
import sys
import os
import time

class convertText(object):
    def __init__(self,fromLangByBaidu,toLangByBaidu,fromLangByMicrosoft,toLangByMicrosoft):
#        self.appid = '20191122000359418'  # 填写你的appid
#        self.secretKey = 'ApHcmTZMFRo65shdLD_h'  # 填写你的密钥  
        self.appid = '20191124000360013'  # 填写你的appid
        self.secretKey = '8TAGPW2Ckk_4_LiLa1Gm'  # 填写你的密钥  
        self.url_baidu_api = 'http://api.fanyi.baidu.com/api/trans/vip/translate' #百度通用api接口
        self.fromLang = fromLangByBaidu
        self.toLang = toLangByBaidu
        self.fromLangByMicrosoft = fromLangByMicrosoft
        self.toLangByMicrosoft = toLangByMicrosoft 
        self.stop_words = self.load_stop_word(os.path.join(sys.path[0], 'stop_words.txt'))

    def _translateFromBaidu(self,text,fromLang,toLang):
        salt = random.randint(32768, 65536)  #随机数
        sign = self.appid + text + str(salt) + self.secretKey   #签名 appid+text+salt+密钥 
        sign = hashlib.md5(sign.encode()).hexdigest()   #sign 的MD5值
        
        url_baidu = self.url_baidu_api + '?appid=' + self.appid + '&q=' + urllib.parse.quote(text) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
#        进程挂起时间1s
#        time.sleep(1)
        
        try:
            response = urllib.request.urlopen(url_baidu,timeout=30)
            content = response.read().decode("utf-8")
            data = json.loads(content)
        
            if 'error_code' in data:
                print('错误代码：{0}, {1}'.format(data['error_code'],data['error_msg']))
                return 'error'
            else:   
                return str(data['trans_result'][0]['dst'])
        except urllib.error.URLError as error:            
            print(error)
            return 'error'
        except urllib.error.HTTPError as error:
            print(error)
            return 'error'
            

    
            
         

#-------test translateFromBaidu-----
#text = '我很喜欢这部电影！你呢？'    
#print(translateFromBaidu(text,fromLang,toLang)) 

    #使用百度翻译API进行回译 chinese->english->chinese
    def convertFromBaidu(self,text):
#        print(self.fromLang,self.toLang)
        translation1 = self._translateFromBaidu(text,self.fromLang,self.toLang)
#        translation1 = self._translateFromBaidu(text,'zh','en')
        if translation1 == 'error':
            return 'error'
        print('1 is over')
        translation2 = self._translateFromBaidu(translation1,self.toLang,self.fromLang)
        if translation2 == 'error':
            return 'error'
        print('2 is over')
#        print(translation1,translation2,text)
        
        if translation2 != text: 
            return translation2

        return 'same'

    #使用微软翻译API进行回译 chinese->english->chinese
    def convertFromMicrosoft(self,text):
        translator1 = Translator(from_lang=self.fromLangByMicrosoft,to_lang=self.toLangByMicrosoft)
        translation1 = translator1.translate(text)
        
        translator2 = Translator(from_lang=self.toLangByMicrosoft,to_lang=self.fromLangByMicrosoft)
        translation2 = translator2.translate(translation1)
        
        if translation2 != text: 
            return translation2
        
        return 'same'
    
    def edaRepalcement(self,text,stop_words,replace_num):
#        中文同义词词典 synonyms 中文近义词工具包，可以用于自然语言理解的很多任务：文本对齐，推荐算法，相似度计算，语义偏移，关键字提取，概念提取，自动摘要，搜索引擎等。
        '''
        随机替换
        '''
        new_words = text.copy()
        random_word_list = list(set([word for word in text if word not in stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0 
        for random_word in random_word_list:
            
            synonym_list = synonyms.nearby(random_word)[0] #返回的是近义词列表 nearby 返回[[近义词],[相似值]]
            
            if len(synonym_list) >= 1:
                synonym = random.choice(synonym_list) #随机选取一个近义词
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
                
            if num_replaced >= replace_num:
                break
        sentence = ' '.join(new_words)
        sentence = sentence.strip()
        new_words = sentence.split(' ')
        
        return new_words #返回的是替换后的词的列表
               
    def _add_words(self,new_words):
        synonym  = []
        count = 0
        while len(synonym) < 1:
            random_word = new_words[random.randint(0,len(new_words)-1)]
            synonym = synonyms.nearby(random_word)[0]
            count += 1
            #如果10次还没有同义词的，就返回
            if count >= 10:
                return
        random_sysnonym = random.choice(synonym)
        random_index = random.randint(0,len(new_words)-1) 
        new_words.insert(random_index,random_sysnonym)
    
    def edaRandomInsert(self,text,insert_num):
        '''
        随机插入
        '''
        new_words = text.copy()
        
        for num in range(insert_num):
            self._add_words(new_words)

        return new_words
    
    def _swap_word(self,new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
            new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words
    
    def edaRandomSwap(self,text,swap_num):
        '''
        随即交换
        '''
        new_words = text.copy()
        for index in range(swap_num):
            new_words = self._swap_word(new_words)
        return new_words
    
    def edaRandomDelete(self,text,p): 
        if len(text) == 1:
            return text
        new_words = []
        for word in text:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
        if len(new_words) == 0:
            rand_int = random.randint(0, len(text)-1)
            return [text[rand_int]]
        return new_words 
    
    def load_stop_word(self,path):
        stop_words = []
        with open(path,'r',encoding ='utf-8') as file:
            for line in file.readlines():
                stop_words.append(line)
        
        return stop_words
    
    def eda(self,text,aug_num,replace_rate,add_rate,swap_rate,delete_rate):
        '''
        默认每种eda方法只使用一次，即产生4条
        '''
        segment_words = jieba.lcut(text)
        num_words = len(segment_words)
        
#        stop_words_path = os.path.join(sys.path[0], 'stop_words.txt')
#        stop_words = self.load_stop_word(stop_words_path)
        stop_words = self.stop_words
       
        replace_num = max(1,int(replace_rate*num_words))
        swap_num = max(1,int(swap_rate*num_words))
        add_num = max(1,int(add_rate*num_words))
        
        text_augment = []
        
        text_replace = ''.join(self.edaRepalcement(segment_words,stop_words,replace_num))
        text_add = ''.join(self.edaRandomInsert(segment_words,add_num))
        text_swap = ''.join(self.edaRandomSwap(segment_words,swap_num))
        text_delete = ''.join(self.edaRandomDelete(segment_words,delete_rate))
        
        
        text_augment.append(text_replace)
        text_augment.append(text_add)
        text_augment.append(text_swap)
        text_augment.append(text_delete)
        
        return text_augment
        
#    def eda_convert(self,data_list):
        
        
        
        
#text = '电影评价,窗前明月光，我很喜十八你欢这部电影！你和啊哈哈呢，我的宝贝？'           
#conText = convertText(fromLangByBaidu='zh',toLangByBaidu='en',fromLangByMicrosoft='chinese',toLangByMicrosoft='english')  
##
#print(conText.convertFromBaidu(text))
#print(conText.convertFromMicrosoft())