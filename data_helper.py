# -*- coding: utf-8 -*-
# @Time : 2019/5/30 下午5:53
# @Author : Honay.King
# @File : data_helper.py
# @Software: PyCharm

import os
import re
import numpy as np
import pandas as pd


def data_clean(sent):
    """
    1. 保留句子中的中文和英文和数字;
    """
    if len(sent) == 0:
        print('[ERROR] data_clean faliled! | The params: {}'.format(sent))
        return None

    sentence = re.sub('[^\u4e00-\u9fa5A-Za-z0-9]', ' ', sent).strip().replace('  ', ' ', 3)

    return sentence


if __name__ == "__main__":

    exit(0)