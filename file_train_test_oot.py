# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 18:21:54 2017

@author: zhanghui
"""

import pandas as pd 
import numpy as np
#import stepwise
from  score_card import Autobinning
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

#import woe



data1 = pd.read_csv('51_bscore2_x_selected.txt', encoding='gbk')

data1=data1.sort_values('lending_time')
data22 = data1.iloc[:, 2:]

#挑选变量
use_var=[]
not_use_var=['contact_relation','installed_apps_version','total_call_in_cnt_1w_vs_3m','user_social_security','platform','lon_contact','total_call_in_cnt_1m_vs_3m','level','phone_loc_contact2_same','total_400_call_cnt_vs_all_ratio']
for col in data22.columns:
    if col not in not_use_var:
        use_var.append(col)
        

data2 =data22[use_var].copy()
    


#样本按时间顺序排序 并按0.7  0.3 划分 traiin+test  oot 
cut1 = int(data2.shape[0]*0.7)

data_tt = data2.iloc[:cut1,:]

data_oot = data2.iloc[cut1+1:,:]

# 样本划分为  train   test 测试
data_train,data_test=train_test_split(data_tt,test_size = 0.3,random_state = 19)



result = Autobinning(data_train, y='target', n=12) #实例化

result.Stepwise(min_iv=0.02) #选择变量

result.bin_to_score() #生成分箱


result.data_to_woe(data_test) #将测试数据换成woe



test_pred = result.model.predict(result.test_data) #预测测试数据


train_auc ,train_ks,train_cut_off = result.compute_auc_ks()
test_auc ,test_ks,test_cut_off = result.compute_auc_ks(result.test_data['target'],test_pred)  #计算ks auc
print('auc_train:',train_auc,'ks_train:',train_ks,'cout_off_train:',train_cut_off)
print('auc_test:',test_auc,'ks_test:',test_ks,'cut_off_train:',test_cut_off)


#跨时间训练
result.data_to_woe(data_oot,oot=True) #将跨时间数据换成woe
oot_pred =result.model.predict(result.oot_data) #预测跨时间数据
oot_auc,oot_ks,oot_cut_off =result.compute_auc_ks(result.oot_data['target'],oot_pred)
print('auc_oot:',oot_auc,'ks_oot:',oot_ks,'cut_off_oot:',oot_cut_off)
result.plot_ks(result.oot_data['target'],oot_pred) #画图 跨时间
result.plot_roc(result.oot_data['target'],oot_pred) #跨时间



result.plot_ks() #画图 训练集
result.plot_ks(result.test_data['target'],test_pred) #画图 测试集

result.plot_roc() #训练集
result.plot_roc(result.test_data['target'],test_pred) #测试集

binning = result.binning #提取分箱
bin_score =result.bin_score #提取分数
iv = result.iv #提取IV

#计算psi
result.PSI()
result.PSI(oot=True)

psi =result.psi #提取测试数据的psi
psi_oot =result.psi_oot #提取跨时间数据的psi
