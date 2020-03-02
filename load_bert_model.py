# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:18:24 2019

@author: dabing
"""

import os 
import sys
import urllib
import zipfile 
from path import DATA_PATH

def get_remote_date(remote_name):   
#    chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
#    chinese_L-12_H-768_A-12
    
    #fileName = os.path.join(sys.path[0],'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip')
    fileName = os.path.join(sys.path[0],'chinese_L-12_H-768_A-12.zip')
    urllib.request.urlretrieve(remote_name, fileName)
    
    #pretrained_dir = os.path.join(sys.path[0],'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16')
    pretrained_dir = os.path.join(sys.path[0],'chinese_L-12_H-768_A-12')
    
    if not os.path.isdir(pretrained_dir):
        os.mkdir(pretrained_dir)
    
    if zipfile.is_zipfile(fileName):
        fz = zipfile.ZipFile(fileName, 'r')

        for file in fz.namelist():
            fz.extract(file, pretrained_dir)
    else:
        print('Cannot load model from', remote_name)
        return None 
    
    #return pretrained_dir
    return os.path.join(pretrained_dir,'chinese_L-12_H-768_A-12')