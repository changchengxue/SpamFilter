#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""

import numpy as np
import AdaboostAndNavieBayes.AdaboostNavieBayes as AdaBoostNB


def get_train_info():
    """
    获取训练算法的DS
    和min_error_rate 的信息
    :return:
    """
    train_ds = np.loadtxt('train_ds.txt', delimiter='\t')
    train_min_error_rate = np.loadtxt('train_min_error_rate.txt', delimiter='\t')
    vocabulary_list = AdaBoostNB.get_vocabulary_list('vocabulary_list.txt')
    prob_words_spam = np.loadtxt('prob_words_spam.txt', delimiter='\t')
    prob_words_health = np.loadtxt('prob_words_health.txt', delimiter='\t')
    prob_spam = np.loadtxt('prob_spam.txt', delimiter='\t')
    return vocabulary_list, prob_words_spam, prob_words_health, prob_spam, train_min_error_rate, train_ds


def simple_test():
    vocabulary_list, prob_words_spam, prob_words_health, prob_spam, train_min_error_rate, train_ds = get_train_info()
    file_name = '../emails/test/test.txt'
    sms_words, class_labels = AdaBoostNB.load_sms_data(file_name)
    test_words_marked_arr = AdaBoostNB.set_of_words_to_vector(vocabulary_list, sms_words[0])
    ps, ph, sms_type = AdaBoostNB.classify(prob_words_spam,
                                           prob_words_health,
                                           train_ds, prob_spam,
                                           test_words_marked_arr)
    print(sms_type)


if __name__ == '__main__':
    simple_test()
