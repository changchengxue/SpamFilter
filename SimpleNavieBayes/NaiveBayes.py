#!/usr/bin/python3.6
# _*_coding: utf-8 _*_

"""
@Author: changcheng
"""

import re
import numpy as np


def text_parser(text):
    """
    对SMS进行预处理
    去掉空字符
    :param text: 输入的字符串
    :return words: 统一小写的List
    """
    # 匹配非字母或数字，即为只留下单词
    reg = re.compile(r'[^a-zA-Z]|\d')
    words = reg.split(text)
    # 去掉字符串，并统一小写
    words = [word.lower() for word in words if len(word) > 0]
    return words


def load_sms_data(file_name):
    """
    加载sms 数据
    class_category: 类别标签， 1 表示是垃圾信息 0 表示是正常信息
    :param file_name: 加载的文件名
    :returns sms_words, class_category:
    """
    file = open(file_name)

    class_category = []
    sms_words = []
    for line in file.readlines():
        line_data = line.strip().split('\t')
        if line_data[0] == 'ham':
            class_category.append(0)
        elif line_data[0] == 'spam':
            class_category.append(1)
        # Slice text
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
    file = open(file_name)
    vocabulary_list = file.readline().strip().split('\t')
    file.close()
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
    return vocabulary_marked


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
    训练数据集中获取语料库中词汇的垃圾信息的概率: P(Wi | S)
    prob_spam: 是垃圾邮件的先验概率P(S)
    :param train_marked_words: 按照语料库标记的数据。二维数组
    :param train_category:
    :return:
    """
    num_train_doc = len(train_marked_words)
    num_words = len(train_marked_words[0])
    prob_spam = sum(train_category) / float(num_train_doc)
    word_in_spm_num = np.ones(num_words)
    word_in_health_num = np.ones(num_words)
    spam_words_num = 2.0
    health_words_num = 2.0
    for i in range(0, num_train_doc):
        if train_category[i] == 1:
            # 如果是垃圾信息
            word_in_spm_num += train_marked_words[i]
            # 统计Spam 中语料库中词汇出现的总次数
            spam_words_num += sum(train_marked_words[i])
        else:
            word_in_health_num += train_marked_words[i]
            health_words_num += sum(train_marked_words[i])
    prob_words_spam = np.log(word_in_spm_num / spam_words_num)
    prob_words_health = np.log(word_in_health_num / health_words_num)

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
    file = open('prop_spam.txt')
    prob_spam = float(file.readline().strip())
    file.close()

    return vocabulary_list, prob_words_spam, prob_words_health, prob_spam


def classify(vocabulary_list, prob_words_spam, prob_words_health, prob_spam, test_words):
    """
    计算联合概率并进行分类
    :param vocabulary_list:
    :param prob_words_spam:
    :param prob_words_health:
    :param prob_spam:
    :param test_words:
    :return:
    """
    test_words_count = set_of_words_to_vector(vocabulary_list, test_words)
    test_words_marked_arr = np.array(test_words_count)
    # 计算 P(Ci|W) W 为向量
    # P(Ci|W) 只需计算P(W|Ci)P(Ci)
    p1 = sum(test_words_marked_arr * prob_words_spam) + np.log(prob_spam)
    p0 = sum(test_words_marked_arr * prob_words_health) + np.log(1 - prob_spam)
    if p1 > p0:
        return 1
    return 0
