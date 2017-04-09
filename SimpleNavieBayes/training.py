#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""

import numpy as np
import SimpleNavieBayes.NaiveBayes as naiveBayes


def main():
    """
    主要的测试流程函数
    """
    file_name = "../emails/training/SMSCollection.txt"
    print("正在生成语料库...\n")
    sms_words, class_labels = naiveBayes.load_sms_data(file_name)
    vocabulary_list = naiveBayes.create_vocabulary_list(sms_words)
    print("生成完成!\n")
    print("正在生成数据标记...\n")
    train_marked_words = naiveBayes.set_of_words_list_to_vector(vocabulary_list, sms_words)
    print("数据标记完成!\n")

    print("正在数据转化...\n")
    # 转成 array 向量
    train_marked_words = np.array(train_marked_words)
    print("数据转成矩阵!\n")

    prob_words_spam, prob_words_health, prob_spam = naiveBayes.training_naive_bayes(
        train_marked_words, class_labels)
    print('prob_spam: ', prob_spam)
    file_spam = open('spm_prob.txt', 'w')
    spam = prob_spam.__str__()
    file_spam.write(spam)
    file_spam.close()
    # 保存训练生成的语料库信息
    # 保存语料库词汇
    file_open = open("vocabulary_list.txt", 'w')
    for i in enumerate(vocabulary_list):
        file_open.write(i[1] + '\t')

    file_open.flush()
    file_open.close()
    # 保存 spam_words_probability, health_words_probability
    np.savetxt('prob_words_spam.txt', prob_words_spam, delimiter='\t')
    np.savetxt('prob_words_health.txt', prob_words_health, delimiter='\t')


if __name__ == '__main__':
    main()
