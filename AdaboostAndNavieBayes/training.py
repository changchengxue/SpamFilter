#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@Author: changcheng
"""

import random
import numpy as np
import AdaboostAndNavieBayes.AdaboostNavieBayes as AdaBoostNB


def training_ada_boost_get_ds(iterate_num=2):
    """
    测试分类的错误率
    :param iterate_num:
    :return:
    """
    file_name = '../emails/training/SMSCollection.txt'
    sms_words, class_labels = AdaBoostNB.load_sms_data(file_name)

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

    # 训练阶段，也可将选择的vocabulary_list 放到这个那个循环中
    # 以选出错误率最低的情况，获取最低错误率的vocabulary_list
    print("正在生成语料库...\n")
    vocabulary_list = AdaBoostNB.create_vocabulary_list(sms_words)
    print("生成语料库!\n")
    print("正在生成数据标记...\n")
    train_marked_words = AdaBoostNB.set_of_words_list_to_vector(vocabulary_list, sms_words)
    print("数据标记完成!\n")
    # 转成array 向量
    print("正在将数据转成矩阵...\n")
    train_marked_words = np.array(train_marked_words)
    print("数据转成矩阵!\n")
    prob_words_spam, prob_words_health, prob_spam = AdaBoostNB.training_naive_bayes(train_marked_words, class_labels)

    DS = np.ones(len(vocabulary_list))

    ds_error_rate = {}
    min_error_rate = np.inf

    for i in range(iterate_num):
        error_count = 0.0
        for j in range(test_count):
            test_words_count = AdaBoostNB.set_of_words_to_vector(vocabulary_list, test_words[j])
            ps, ph, sms_type = AdaBoostNB.classify(prob_words_spam, prob_words_health, DS, prob_spam, test_words_count)
            if sms_type != test_words_type[j]:
                error_count += 1
                # alpha = (ph - ps) / ps
                alpha = ps - ph
                # if alpha < 0
                # 原先为spam，预测为ham，则ERROR～
                if alpha > 0:
                    # 原先为ham，预测成spam
                    DS[test_words_count != 0] = np.abs(
                        (DS[test_words_count != 0] - np.exp(alpha)) / DS[test_words_count != 0])
                else:
                    DS[test_words_count != 0] = (DS[test_words_count != 0] + np.exp(alpha)) / DS[test_words_count != 0]
        print('DS: ', DS)
        error_rate = error_count / test_count
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            ds_error_rate['min_error_rate'] = min_error_rate
            ds_error_rate['DS'] = DS
        print("第 %d 轮迭代， 错误个数为 %d， 错误率为 %f " % (i+1, error_count, error_rate))

        if error_rate == 0.0:
            break

    ds_error_rate['vocabulary_list'] = vocabulary_list
    ds_error_rate['prob_words_spam'] = prob_words_spam
    ds_error_rate['prob_words_health'] = prob_words_health
    ds_error_rate['prob_spam'] = prob_spam
    return ds_error_rate


if __name__ == '__main__':
    # iter_num = input("需要迭代多少次：")
    # DS_error_rate = training_ada_boost_get_ds(int(iter_num))
    DS_error_rate = training_ada_boost_get_ds()
    # 保存信息
    np.savetxt('prob_words_spam.txt', np.array(DS_error_rate['prob_words_spam']), delimiter='\t')
    np.savetxt('prob_words_health.txt', np.array(DS_error_rate['prob_words_health']), delimiter='\t')
    save_prob_spam = open('prob_spam.txt', 'w')
    save_prob_spam.write(str(DS_error_rate['prob_spam']))
    # np.savetxt('prob_spam.txt', np.array(DS_error_rate['prob_spam']), delimiter='\t')
    np.savetxt('train_ds.txt', np.array(DS_error_rate['DS']), delimiter='\t')
    np.savetxt('train_min_error_rate.txt', np.array([DS_error_rate['min_error_rate']]), delimiter='\t')
    vocabulary = DS_error_rate['vocabulary_list']
    file_write = open('vocabulary_list.txt', 'w')
    for i in range(len(vocabulary)):
        file_write.write(vocabulary[i] + '\t')
    file_write.flush()
    file_write.close()
