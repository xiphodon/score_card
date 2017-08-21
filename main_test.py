#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/31 17:18
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : main_test.py
# @Software: PyCharm

from sklearn.model_selection import train_test_split
from score_card import Autobinning
import pandas as pd
import time

if __name__ == "__main__":

    data1 = pd.read_csv(r'input/data_baidu_phone_qq5.csv', encoding='utf8')

    # 挑选变量
    use_var = []
    not_use_var = ["id"]

    # for col in data1.columns:
    #     if col not in not_use_var:
    #         use_var.append(col)
    use_var = list(set(data1.columns) - set(not_use_var)) # ++++++++++++++++++++++++++++++change 1

    data2 = data1[use_var].copy()

    print("原数据集 shape ：", data2.shape)


    # 样本按时间顺序排序 并按0.7  0.3 划分 traiin+test  oot
    cut1 = int(data2.shape[0] * 0.7)

    data_tt = data2.iloc[:cut1, :]

    data_oot = data2.iloc[cut1 + 1:, :]

    # 样本划分为  train   test 测试
    data_train, data_test = train_test_split(data_tt, test_size=0.3, random_state=19)

    print("训练集 shape ：", data_train.shape)
    print("测试集 shape ：", data_test.shape)
    print("oot验证集 shape ：", data_oot.shape)

    result = Autobinning(data_train, y='overdue', n=12)  # 实例化

    # result.Stepwise(min_iv=0.02)  # 选择变量
    #
    # result.bin_to_score()  # 生成分箱
    #
    # result.data_to_woe(data_test)  # 将测试数据换成woe
    #
    # test_pred = result.model.predict(result.test_data)  # 预测测试数据
    #
    # train_auc, train_ks, train_cut_off = result.compute_auc_ks()
    # test_auc, test_ks, test_cut_off = result.compute_auc_ks(result.test_data['overdue'], test_pred)  # 计算ks auc
    # print('auc_train:', train_auc, 'ks_train:', train_ks, 'cout_off_train:', train_cut_off)
    # print('auc_test:', test_auc, 'ks_test:', test_ks, 'cut_off_train:', test_cut_off)

    ###################################################################################################

    # # 跨时间训练
    # result.data_to_woe(data_oot, oot=True)  # 将跨时间数据换成woe
    # oot_pred = result.model.predict(result.oot_data)  # 预测跨时间数据
    # oot_auc, oot_ks, oot_cut_off = result.compute_auc_ks(result.oot_data['target'], oot_pred)
    # print('auc_oot:', oot_auc, 'ks_oot:', oot_ks, 'cut_off_oot:', oot_cut_off)
    # result.plot_ks(result.oot_data['target'], oot_pred)  # 画图 跨时间
    # result.plot_roc(result.oot_data['target'], oot_pred)  # 跨时间
    #
    # result.plot_ks()  # 画图 训练集
    # result.plot_ks(result.test_data['target'], test_pred)  # 画图 测试集
    #
    # result.plot_roc()  # 训练集
    # result.plot_roc(result.test_data['target'], test_pred)  # 测试集
    #
    # binning = result.binning  # 提取分箱
    # bin_score = result.bin_score  # 提取分数
    # iv = result.iv  # 提取IV
    #
    # # 计算psi
    # result.PSI()
    # result.PSI(oot=True)
    #
    # psi = result.psi  # 提取测试数据的psi
    # psi_oot = result.psi_oot  # 提取跨时间数据的psi

