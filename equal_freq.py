# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:57:23 2017

@author: zhanghui
"""

import pandas as pd 

def Equal_Freq(data,x,y,n):
    '''
    等频分箱
    :param data: 训练集
    :param x: 需要分箱的列名
    :param y: 标签名
    :param n: 分箱数
    :return:
    '''
    data_copy = data[[x,y]].copy() # [分箱列,标签列]备份
    # data_new = data[[x,y]][pd.isnull(data[x])==False].copy()
    data_new = data[[x,y]][pd.notnull(data[x])].copy() # 非空[分箱列,标签列]数据集

    data_gb = data_new.groupby(x)
    data_cnt = data_gb.count() # 统计分箱列各个值得频率 df

    data_cnt['F'] = data_cnt.cumsum() # 向前求和，目前data_cnt 有两列，索引为x列的值，overdue列为对应index频率，F为赔率累计求和数
    avg_freq = max(data_cnt['F'])/n # 总频数/n = 平均频数
    cut_points=['-inf']
    for i in range(1,n):        
        cut_points.append(abs(data_cnt['F']-avg_freq*i).argmin())
    cut_points.append('inf') # 负无穷和正无穷的分箱切点
    
    cut_points_n=len(cut_points)

    for j in range(1,cut_points_n) :
        if j == 1 :
            left= float(cut_points.pop(0))
            #print(cut_points,x,'dddd')
            right = float(cut_points.pop(0))
        else :
            left = right
            #print(cut_points,x,'sss')

            right = float(cut_points.pop(0))
    
        interval = pd.Interval(left,right)

        data_copy[x].where(data_copy[x].apply(lambda x: (x <= left or x > right ) if (isinstance(x,float) or isinstance(x,int) ) and pd.isnull(x)==False else True) ,interval,inplace = True )      
        
    return data_copy[x]
    
        
    
    