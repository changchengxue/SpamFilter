#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""

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


def load_sms_data(file_name):
    """
    加载SMS 数据
    """
    file_open = open(file_name)
    class_category = []
    # 类别，1表示垃圾信息 0 表示正常
    sms_words = []
    for line in file_open.readlines():
        line_data = line.strip().split('\t')
        if line_data[0] == 'ham':
            class_category.append(0)
        else:
            class_category.append(1)
        # 切分文本
        words = text_parser(line_data[1])
        sms_words.append(words)
    return sms_words, class_category


def create_vocabulary_list(sms_words):
    """
    创建语料库
    """
    vocabulary_set = set([])
    for words in sms_words:
        vocabulary_set = vocabulary_set | set(words)
    vocabulary_list = list(vocabulary_set)
    return vocabulary_list


def set_of_words_to_vector(vocabulary_list, sms_words):
    """
    sms 内容匹配语料库
    表示语料库的词汇出现的次数
    """
    vocabulary_marked = [0] * len(vocabulary_list)
    for sms_word in sms_words:
        if sms_word in vocabulary_list:
            vocabulary_marked[vocabulary_list.index(sms_word)] = 1
    return vocabulary_marked


def training_naive_bayes(train_marked_words, train_category):
    """
    训练数据集中获取语料库中词汇的 spam
    P(Wi|S)
    :param train_marked_words: 按照语料库标记的数据，二维数组
    :param train_category:
    """
    num_train_doc = len(train_marked_words)
    num_words = len(train_marked_words[0])
    # 是垃圾邮件的先验概率P(S)
    spam_probability = sum(train_category) / float(num_train_doc)
    # 统计语料库中词汇在S和H 中出现的次数
    words_in_spam_num = np.zeros((1, num_words))
    words_in_health_num = np.zeros((1, num_words))
    spam_words_num = 0.0
    health_words_num = 0.0
    for i in range(0, num_train_doc):
        if train_category[i] == 1:
            # 如果是垃圾邮件
            words_in_spam_num += train_marked_words[i]
            # 统计Spam 中语料库中词汇出现的总次数
            spam_words_num += sum(train_marked_words[i])
        else:
            words_in_health_num += train_marked_words[i]
            health_words_num += sum(train_marked_words[i])

    # 计算语料库中词汇概率 P(Wi | S) P(Wi | H)
    p_words_spam = words_in_spam_num / spam_words_num
    p_words_health = words_in_health_num / health_words_num

    return p_words_spam, p_words_health, spam_probability


def bayes_theorem_cal_prob(spam_words_probability, health_words_probability, spam_probability):
    """
    利用Bayes 定理计算P(S|Wi)
    即词汇Wi 出现，是垃圾邮件的条件概率
                    P(Wi|S)P(S)
    P(S|Wi) = -----------------------------
                P(Wi|S)P(S) + P(Wi|H)P(H)
    """
    temp = spam_words_probability  * spam_probability
    wi_probability = temp / (temp + health_words_probability * (1 - spam_probability))
    return wi_probability
