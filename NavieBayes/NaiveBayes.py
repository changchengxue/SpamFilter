#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import re
import numpy as np


def text_parser(text):
    """
    对SMS 进行预处理，去除空字符串，并统一小写
    """
    reg_ex = re.compile(r'[^a-zA-Z]|\d')
    # 匹配非字母或数字
    # 即只留下单词
    words = reg_ex.split(text)
    words = [word.lower() for word in words if len(word) > 0]
    # 去除空字符串，并统一小写
    return words
