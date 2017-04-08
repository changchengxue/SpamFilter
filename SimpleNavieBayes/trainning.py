#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""

import numpy as np
import NaiveBayes as util


def main():
    """
    主要的测试流程函数
    """
    file_name = "../emails/training/SMSCollection.txt"
    print("正在生成语料库...\n")
    sms_words, class_labels = util.load_sms_data(file_name)
    vocabulary_list = util.create_vocabulary_list(sms_words)
    print("生成完成!\n")
    train_marked_words = []
    print("正在生成数据标记...\n")
    for words in sms_words:
        vocabulary_marked = util.set_of_words_to_vector(vocabulary_list, words)
        train_marked_words.append(vocabulary_marked)
    print("数据标记完成!\n")

    print("正在数据转化...\n")
    # 转成 array 向量
    train_marked_words = np.array(train_marked_words)
    print("数据转成矩阵!\n")

    spam_words_probability, health_words_probability, spam_probability = util.training_naive_bayes(
        train_marked_words, class_labels)
    print("垃圾邮件的概率: ", spam_probability)

    # 保存训练生成的语料库信息
    # 保存语料库词汇
    file_opne = open("vocabulary_list.txt", 'w')
    for i in range(len(vocabulary_list)):
        file_opne.write(vocabulary_list[i] + '\t')

    # 保存 spam_wrods_probability, health_words_probability
    np.savetxt('spam_words_probability.txt', spam_words_probability, delimiter='\t')
    np.savetxt('health_words_probability.txt', health_words_probability, delimiter='\t')


if __name__ == '__main__':
    main()
