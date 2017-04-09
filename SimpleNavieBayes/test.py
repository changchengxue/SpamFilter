#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""

import random
import numpy as np
import SimpleNavieBayes.NaiveBayes as naiveBayes


def simple_test():
    """
    测试函数
    :return:
    """
    vocabulary_list, prob_words_spam, prob_words_health, prob_spam = \
        naiveBayes.get_trained_model_info()
    file_name = '../emails/test/test.txt'
    sms_words, class_labels = naiveBayes.load_sms_data(file_name)
    sms_type = naiveBayes.classify(vocabulary_list, prob_words_spam,
                                   prob_words_health, prob_spam, sms_words[0])
    print(sms_type)


def test_classify_error_rate():
    """

    :return:
    """
    file_name = '../emails/training/SMSCollection.txt'
    sms_words, class_labels = naiveBayes.load_sms_data(file_name)
    # 交叉检验
    test_words = []
    test_words_type = []

    test_count = 1000
    for i in range(test_count):
        random_index = int(random.uniform(0, len(sms_words)))
        test_words_type.append(class_labels[random_index])
        test_words.append(sms_words[random_index])
        del (sms_words[random_index])
        del (class_labels[random_index])

    vocabulary_list = naiveBayes.create_vocabulary_list(sms_words)
    print("生成语料库...\n")
    train_marked_words = naiveBayes.set_of_words_list_to_vector(vocabulary_list, sms_words)
    print("数据标记完成!\n")
    # 转成 array 向量
    train_marked_words = np.array(train_marked_words)
    print("数据转成矩阵!\n")
    prob_words_spam, prob_words_health, prob_spam = naiveBayes.training_naive_bayes(
        train_marked_words, class_labels)

    error_count = 0.0
    for i in range(test_count):
        sms_type = naiveBayes.classify(
            vocabulary_list, prob_words_spam, prob_words_health, prob_spam, test_words[i])
        print("预测类别: ", sms_type, "实际类别: ", test_words_type[i])
        if sms_type != test_words_type[i]:
            error_count += 1

    print("错误个数: ", error_count, "错误率: ", error_count/test_count)


if __name__ == '__main__':
    test_classify_error_rate()
