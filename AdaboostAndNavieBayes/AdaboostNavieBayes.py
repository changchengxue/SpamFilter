#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""
import numpy as np


def text_parser(text):
    """
    对SMS 进行预处理
    去掉字符串
    并统一小写
    :param text:
    :return:
    """
    import re
    reg = re.compile(r'[^a-zA-Z]|\d')
    words = reg.split(text)
    words = [word.lower() for word in words if len(word) > 0]
    return words


def load_sms_data(file_name):
    """
    加载sms 数据
    :param file_name:
    :return:
    """
    file = open(file_name)
    class_category = []
    # 类别标签，1表示是垃圾SMS，0表示正常SMS
    sms_words = []
    for line in file.readlines():
        line_data = line.strip().split('\t')
        if line_data[0] == 'ham':
            class_category.append(0)
        elif line_data[0] == 'spam':
            class_category.append(1)
        # 切分文本
        words = text_parser(line_data[1])
        sms_words.append(words)
    return sms_words, class_category


def create_vocabulary_list(sms_words):
    """
    创建语料库
    :param sms_words:
    :return:
    """
    vocabulary_set = set([])
    for words in sms_words:
        vocabulary_set |= set(words)
    vocabulary_list = list(vocabulary_set)
    return vocabulary_list


def get_vocabulary_list(file_name):
    """
    从词汇列表文件中获取语料库
    :param file_name:
    :return:
    """
    file_read = open(file_name)
    vocabulary_list = file_read.readline().strip().split('\t')
    file_read.close()
    return vocabulary_list


def set_of_words_to_vector(vocabulary_list, sms_words):
    """
    sms 内容匹配语料库
    标记语料库的词汇出现的次数
    :param vocabulary_list:
    :param sms_words:
    :return:
    """
    vocabulary_marked = [0] * len(vocabulary_list)
    for sms_word in sms_words:
        if sms_word in vocabulary_list:
            vocabulary_marked[vocabulary_list.index(sms_word)] += 1
    return np.array(vocabulary_marked)


def set_of_words_list_to_vector(vocabulary_list, sms_words_list):
    """
    将文本数据的二维数组标记
    :param vocabulary_list:
    :param sms_words_list:
    :return:
    """
    vocabulary_marked_list = []
    for i in range(len(sms_words_list)):
        vocabulary_marked = set_of_words_to_vector(vocabulary_list, sms_words_list[i])
        vocabulary_marked_list.append(vocabulary_marked)
    return vocabulary_marked_list


def training_naive_bayes(train_marked_words, train_category):
    """
    训练数据集中获取语料库中词汇的spam P(Wi|S)
    :param train_marked_words:
    :param train_category:
    :return:
    """
    num_train_doc = len(train_marked_words)
    num_words = len(train_marked_words[0])
    prob_spam = sum(train_category) / float(num_train_doc)

    words_in_spam_num = np.ones(num_words)
    words_in_health_num = np.ones(num_words)

    spam_words_num = 2.0
    health_words_num = 2.0

    for i in range(0, num_train_doc):
        if train_category[i] == 1:
            words_in_spam_num += train_marked_words[i]
            spam_words_num += sum(train_marked_words[i])
        else:
            words_in_health_num += train_marked_words[i]
            health_words_num += sum(train_marked_words[i])

    prob_words_spam = np.log(words_in_spam_num / spam_words_num)
    prob_words_health = np.log(words_in_health_num / health_words_num)

    return prob_words_spam, prob_words_health, prob_spam


def get_trained_model_info():
    """
    获取训练的模型信息
    :return:
    """
    # 加载训练获取的语料库信息
    vocabulary_list = get_vocabulary_list('vocabulary_list.txt')
    prob_words_health = np.loadtxt('prob_words_health.txt', delimiter='\t')
    prob_words_spam = np.loadtxt('prob_words_spam.txt', delimiter='\t')
    file = open('prob_spam.txt')
    prob_spam = float(file.readline().strip())
    file.close()

    return vocabulary_list, prob_words_spam, prob_words_health, prob_spam


def classify(prob_words_spam, prob_words_health, DS, prob_spam, test_words_marked_arr):
    """
    计算联合概率进行分类
    :param prob_words_spam:
    :param prob_words_health:
    :param DS: Adaboost 算法额外增加的权重系数
    :param prob_spam:
    :param test_words_marked_arr:
    :return:
    """
    # 计算P(Ci|W)
    # W 为向量
    # P(Ci|W) 只需计算P(W|Ci) P(Ci)
    ps = sum(test_words_marked_arr * prob_words_spam * DS) + np.log(prob_spam)
    ph = sum(test_words_marked_arr * prob_words_health) + np.log(1-prob_spam)
    if ps > ph:
        return ps, ph, 1
    else:
        return ps, ph, 0
