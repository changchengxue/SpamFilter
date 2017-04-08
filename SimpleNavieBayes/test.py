#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""

import numpy as np
import NaiveBayes as nb


VOCABULARY_LIST = nb.create_vocabulary_list('vocabulary_list.txt')
HEALTH = np.loadtxt('health_words_probability.txt', delimiter='\t')
SPAM = np.loadtxt('spam_words_probability.txt', delimiter='\t')


def main():
    """
    Test function
    """
    file_read = open('spam_prob.txt')
    spam_prop = float(file_read.readline().strip())
    print('Spam probility:', type(spam_prop), spam_prop)
    spam_wi_prop, health_wi_prop = nb.bayes_theorem_cal_prob(
        SPAM, HEALTH, spam_prop)
    file_name = '../emails/test/test.txt'
    sms_words, class_lables = nb.load_sms_data(file_name)
    test_words_count = nb.set_of_words_to_vector(VOCABULARY_LIST, sms_words[0])
    test_words_count_arr = np.array(test_words_count)
    result = nb.classify(spam_wi_prop, health_wi_prop, spam_prop, test_words_count_arr)
    print('result: ', result)

if __name__ == '__main__':
    main()
