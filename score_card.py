# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:12:03 2017

creditscorecard_python

@author: zhanghui
"""



import pandas as pd 
import numpy as np
import math
from scipy.stats.stats import pearsonr
from equal_freq import Equal_Freq
import statsmodels.formula.api as smf
#import sklearn.model_selection as slc  
import sklearn.metrics as smt
import matplotlib.pyplot as plt
from numpy.linalg.linalg import LinAlgError


class Autobinning :
    """

    dataParameters:
    --------------
    data:pandas DataFrame with all possible predictors and response
    
    y:string,name of y column in data
    
    n:continous variable max bin
    
    
    Returns:
    -------------
    woe_x: data replaced with woe , is a dict
    
    binning: bin details for each valid variable ,is a dict
    
    iv: iv values for each valid variable ,is a dict
               


    """

        
    def __init__(self,data,y,n=20):
        '''

        :param data: 训练集
        :param y: 标签名
        :param n: 最大分箱数
        '''
        
        np.seterr(divide='ignore', invalid='ignore') # 可以接受分母无限接近0的情况

        col_number=[]
        col_charac=[]
        need_change_var = []
        for col in data.columns :
            if col!=y :
                if (data[col].dtype =='float64' or data[col].dtype== 'int64') and len(data[col].unique()) >1 :
                    # 认为是连续变量
                    col_number.append(col)
                if (data[col].dtype !='float64' and data[col].dtype != 'int64') and len(data[col].unique())>1:
                    # 认为是名义变量
                    col_charac.append(col)

        data_number= data.loc[:,col_number]
        data_charac= data.loc[:,col_charac]
        label =  data[y]
        
        
        final_bin={}
        final_iv={}
        data_bin={}
        data_bin[y]=data[y]
        tot_good = label.count()-label.sum()
        tot_bad= label.sum()
        
        print('continous variables : \n ',col_number,'\n @@@@@@@@@@@ \n','caractor variables : \n',col_charac)
        
        
        if data_number.empty == False :
         
            #对所有连续变量分箱
            for colname in data_number.columns:
               
                #等频n次尝试，明细数据结果保存在data_bins
                data_bins=pd.DataFrame( data[y])

                try: 
                    for i in range(2,n):
                        data_bins[i]=Equal_Freq(data,colname,y,i) #等频分箱 尝试 2~n 种分箱
                except Exception as e:
                    print(e)
                    print('!!!!!!!!!!!!!!!\n',colname,'  need change to charactor\n','!!!!!!!!!!!!!!\n ')
                    need_change_var.append(colname)
                    continue
                    
                data_bins_desc = data_bins.copy()
                     
                
                final_bins_incr=[]
                final_bins_desc=[]
                final_ivs_incr=[]
                final_ivs_desc=[]
                final_bin_incr={}
                final_iv_incr={}
                data_bin_incr={}
                final_bin_desc={}
                final_iv_desc={}
                data_bin_desc={}
                
                data_pearsonr = data[[colname,y]][pd.isnull(data[colname])==False] # 非空分箱列名、标签列名、数据集
                
                if pd.isnull(pearsonr(data_pearsonr.iloc[:,0],data_pearsonr.iloc[:,1])[0]):
                    print('!!!!!!!!!!!!!!!\n',colname,'  need change to charactor\n','!!!!!!!!!!!!!!\n ')
                    need_change_var.append(colname)
                    continue                    
                
                if pearsonr(data_pearsonr.iloc[:,0],data_pearsonr.iloc[:,1])[0]<0 : #判断该变量与y的相关性，若为负相关则采用单调递增分箱
     
                    for m in range(2,n):
                            min_idx={}  #单调递增
                            ######## matlab  MAPA算法 开始
                            for j in range(2,n):
                            
                                bin_count = pd.crosstab(index=data_bins[m],columns=data_bins[y])
                                if 1 not in bin_count.columns: # 分箱中没有target=1的情况
                                    bin_count[1] = 0
                                if 0 not in bin_count.columns: # 分箱中没有target=0的情况
                                    bin_count[0] = 0
                                new_index=[]
                                for index in bin_count.index:
                                        if index not in min_idx:
                                            new_index.append(index)
                                if len(new_index) ==0 : # 若只有一组 跳出循环
                                    break
                                bin_count_new=bin_count.loc[new_index,].copy()             
                                bin_count_cum = bin_count_new.cumsum()
                                bin_count_cum['good%']=bin_count_cum[0]/(bin_count_cum[0]+bin_count_cum[1])
                                min_idx1 = bin_count_cum.iloc[:,2].argmin()
                                if pd.isnull(min_idx1) :
                                    min_loc=0
                                else :
                                    min_loc = list(bin_count_cum.index).index(min_idx1)
                                if min_loc == 0 :
                                    min_idx[min_idx1]=min_idx1
                                    
                                else:#给出应合并的左右值
                                    a=bin_count_cum.index[0].left
                                    b=bin_count_cum.index[min_loc].right
                                    interval_new=pd.Interval(a,b)
                                    for x in  np.arange(min_loc+1):#对应更改明细数据分箱
                                        index_1 = bin_count_cum.index[x]
                                        data_bins[m].where(data_bins[m]!=index_1,interval_new,inplace=True)
                                    min_idx[interval_new]=interval_new

                            ######## matlab  MAPA算法 结束
                                
                            # 计算单调分箱后各组的WOE和IV
                            bin_count_fil =  pd.crosstab(index=data_bins[m],columns=data_bins[y])
                            
                            if 1 not in bin_count_fil.columns:
                                    bin_count_fil[1] = 0
                            if 0 not in bin_count_fil.columns:
                                    bin_count_fil[0] = 0
                            #print(bin_count_fil,'$$$$$')
                            bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf'))
                            bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                            #判断是否有nan,并且nan的woe是否inf,若'是'则做以下处理：
        
                            data_nan=data_bins.loc[:,[y,m]][pd.isnull(data_bins[m])]
         
                            if data_nan.empty == False:#若有NAN值
                                if len(data_nan[y].unique()) >1 :#判断NAN值是否有好有坏
                                    data_nan['count']=1
                                    data_nan_cnt=data_nan.groupby(y).count()
                                else:
                                    if data_nan.iloc[0,0]==0:
                                        data_nan_cnt=pd.DataFrame([[0,list(data_nan.shape)[0]],[0,0]],index=[0.0, 1.0],columns=[m,'count'])
                                    else:
                                        data_nan_cnt=pd.DataFrame([[0,0],[0,list(data_nan.shape)[0]]],index=[0.0, 1.0],columns=[m,'count'])
        
                                woe_na = np.log((data_nan_cnt.loc[0,'count']/data_nan_cnt.loc[1,'count'])/(tot_good/tot_bad))
                                data_na_woe=pd.DataFrame([float('NaN')],index=[0],columns=['bin'])
                                data_na_woe[0]=data_nan_cnt.loc[0,'count']
                                data_na_woe[1]=data_nan_cnt.loc[1,'count']
                            else : # 若没有NAN值 则woe_na 置为 nan
                                data_na_woe=pd.DataFrame([float('NaN')],index=[0],columns=['bin'])
                                data_na_woe[0]=0
                                data_na_woe[1]=0
                                woe_na = float('NaN')
                            if  np.isinf(woe_na) : #判断缺失值的Woe是否正负无穷
                                iv_na = 0
                                if (woe_na>0 and bin_count_fil.iloc[0,2]>bin_count_fil.iloc[len(bin_count_fil)-1,2]) or ( woe_na<0 and bin_count_fil.iloc[0,2]<bin_count_fil.iloc[len(bin_count_fil)-1,2]):
                                    bin_count_fil.iloc[0,0] = bin_count_fil.iloc[0,0]+data_nan_cnt.loc[0,'count']
                                    bin_count_fil.iloc[0,1] = bin_count_fil.iloc[0,1]+data_nan_cnt.loc[1,'count']
                                    bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf'))
                                    bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                                    woe_na=  bin_count_fil.iloc[0,2]
                                    data_na_woe[0] =bin_count_fil.iloc[0,0]
                                    data_na_woe[1] =bin_count_fil.iloc[0,1]
                                else :
                                    bin_count_fil.iloc[len(bin_count_fil)-1,0] = bin_count_fil.iloc[len(bin_count_fil)-1,0]+data_nan_cnt.loc[0,'count']
                                    bin_count_fil.iloc[len(bin_count_fil)-1,1] = bin_count_fil.iloc[len(bin_count_fil)-1,1]+data_nan_cnt.loc[1,'count']
                                    bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf'))
                                    bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                                    woe_na=  bin_count_fil.iloc[len(bin_count_fil)-1,2]
                                    data_na_woe[0]=  bin_count_fil.iloc[len(bin_count_fil)-1,0] 
                                    data_na_woe[1]=  bin_count_fil.iloc[len(bin_count_fil)-1,1] 
        
                            else :
                                iv_na = ((data_na_woe[0]/tot_good)-(data_na_woe[1]/tot_bad))*woe_na
        
                            data_na_woe['woe']=woe_na
                            data_na_woe['iv'] = iv_na 
                            for jj in np.arange(len(bin_count_fil)) :#判断分箱顶端是否足够量并且不为正负无穷
                                if (bin_count_fil.iloc[0,0]+bin_count_fil.iloc[0,1])>=(tot_good+tot_bad)*0.05 and np.isinf(bin_count_fil.iloc[0,2])==False :
                                    break
                                else:
                                    if len(bin_count_fil)==1 :
                                        bin_count_fil.iloc[0,0]=tot_good
                                        bin_count_fil.iloc[0,1]=tot_bad
                                        bin_count_fil.iloc[0,2]=0
                                        data_na_woe['woe']=0
                                        data_na_woe['iv']=0
                                    else:
                                        if bin_count_fil.iloc[1,2] > data_na_woe['woe'].values and data_na_woe['woe'].values != bin_count_fil.iloc[0,2] :
                                            bin_count_fil.iloc[0,0]=bin_count_fil.iloc[0,0]+ data_na_woe[0].values
                                            bin_count_fil.iloc[0,1]=bin_count_fil.iloc[0,1] + data_na_woe[1].values
                                            bin_count_fil.iloc[0,2] = math.log(((bin_count_fil.iloc[0,0]/bin_count_fil.iloc[0,1])/(tot_good/tot_bad)))   if round(((bin_count_fil.iloc[0,0]/bin_count_fil.iloc[0,1])/(tot_good/tot_bad)),100)!=0 else float('-inf')
                                            data_na_woe['woe'] = bin_count_fil.iloc[0,2]
                                            data_na_woe['iv'] =0 
                                                
                                        else :
                                            left =bin_count_fil.index[0].left
                                            right = bin_count_fil.index[1].right
                                            interval_new2 = pd.Interval(left,right)
                                            index0 = bin_count_fil.index[0]
                                            index1 = bin_count_fil.index[1]
                                            data_bins[m].where(data_bins[m]!=index0,interval_new2,inplace=True)
                                            data_bins[m].where(data_bins[m]!=index1,interval_new2,inplace=True)
                                            bin_count_fil =  pd.crosstab(index=data_bins[m],columns=data_bins[y])
                                            if  1 not in bin_count_fil.columns:
                                                bin_count_fil[1]=0
                                            if 0 not in bin_count_fil.columns:
                                                bin_count_fil[0]=0
                                            bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf') )
                                            bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                                    

                            for jj in np.arange(len(bin_count_fil)) :#判断分箱底端是否足够量并且不为正负无穷
                                #print(bin_count_fil,'######',bin_count_fil.columns)
                                if (bin_count_fil.iloc[len(bin_count_fil)-1,0]+bin_count_fil.iloc[len(bin_count_fil)-1,1])>=(tot_good+tot_bad)*0.05  and np.isinf(bin_count_fil.iloc[len(bin_count_fil)-1,2])==False :
                                    break
                                else: 
                                    if len(bin_count_fil)==1 :
                                        bin_count_fil.iloc[0,0]=tot_good
                                        bin_count_fil.iloc[0,1]=tot_bad
                                        bin_count_fil.iloc[0,2]=0
                                        data_na_woe['woe']=0
                                        data_na_woe['iv']=0
                                    else:
                                        if bin_count_fil.iloc[len(bin_count_fil)-2,2]  < data_na_woe['woe'].values and data_na_woe['woe'].values != bin_count_fil.iloc[len(bin_count_fil)-1,2]  :
                                            bin_count_fil.iloc[len(bin_count_fil)-1,0]=bin_count_fil.iloc[len(bin_count_fil)-1,0]+ data_na_woe[0].values
                                            bin_count_fil.iloc[len(bin_count_fil)-1,1]=bin_count_fil.iloc[len(bin_count_fil)-1,1] + data_na_woe[1].values
                                            bin_count_fil.iloc[len(bin_count_fil)-1,2] = math.log(((bin_count_fil.iloc[len(bin_count_fil)-1,0]/bin_count_fil.iloc[len(bin_count_fil)-1,1])/(tot_good/tot_bad)))   if round(((bin_count_fil.iloc[len(bin_count_fil)-1,0]/bin_count_fil.iloc[len(bin_count_fil)-1,1])/(tot_good/tot_bad)),100) !=0 else float('-inf')
                                            data_na_woe['woe'] = bin_count_fil.iloc[len(bin_count_fil)-1,2]
                                            data_na_woe['iv'] =0 
                                                
                                        else :
                                            left =bin_count_fil.index[len(bin_count_fil)-2].left
                                            right = bin_count_fil.index[len(bin_count_fil)-1].right
                                            interval_new2 = pd.Interval(left,right)
                                            index0 = bin_count_fil.index[len(bin_count_fil)-2]
                                            index1 = bin_count_fil.index[len(bin_count_fil)-1]
                                            data_bins[m].where(data_bins[m]!=index0,interval_new2,inplace=True)
                                            data_bins[m].where(data_bins[m]!=index1,interval_new2,inplace=True)
                                            bin_count_fil =  pd.crosstab(index=data_bins[m],columns=data_bins[y])
                                            if 1 not in bin_count_fil.columns:
                                                bin_count_fil[1]=0
                                            if 0 not in bin_count_fil.columns:
                                                bin_count_fil[0]=1
                                            bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf') )
                                            bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                                         
                            if len(bin_count_fil)<2 and  np.isinf(bin_count_fil['woe'].values) :
                                bin_count_fil[0]=tot_good
                                bin_count_fil[1]=tot_bad
                                bin_count_fil['woe']=0
                                bin_count_fil['iv'] =0
                                data_na_woe[0]=tot_good
                                data_na_woe[1]=tot_bad
                                data_na_woe['woe']=0
                                data_na_woe['iv']=0
                            
                            #合并nan的woe iv
        
                            bin_count_fil['bin']=bin_count_fil.index
                            bin_count_fil=bin_count_fil.reindex(columns=['bin',0.0, 1.0, 'woe', 'iv'])
                            bin_count_final = pd.concat([bin_count_fil,data_na_woe],ignore_index=True)
                            tot_iv=bin_count_final['iv'].sum()
                            final_bins_incr.append(bin_count_final)
                            final_ivs_incr.append(tot_iv)
                        
                        #选出所有单调递增分箱尝试中最佳的一种分箱
                    max_iv_idx_incr = final_ivs_incr.index(max(final_ivs_incr))
                    final_bin_incr[colname]=final_bins_incr[max_iv_idx_incr]
                    final_iv_incr[colname]=max(final_ivs_incr)
                    data_bin_opt = data_bins[max_iv_idx_incr+2]
                    for idx in final_bin_incr[colname]['bin']:  #将明细数据中的分箱替换成woe
                        data_bin_opt.where(data_bin_opt!=idx,final_bin_incr[colname][final_bin_incr[colname]['bin']==idx]['woe'].values,inplace=True)
                         #替换缺失值
                       
                        na_w = final_bin_incr[colname][pd.isnull(final_bin_incr[colname]['bin'])]['woe'].values 
                        if pd.isnull(na_w) ==False :
                        
                            data_bin_opt.where(pd.isnull(data_bin_opt)==False,na_w,inplace=True)               
             
                    if len(data_bin_incr)==0:
                        data_bin_incr['y']=data_bins[y]
                    data_bin_incr[colname] =data_bin_opt
                    
               
    
                else :
    
                    #单调递减
                    for i in range(2,n):
                            max_idx={}  #单调递增的尝试
                            for j in range(2,n):
                                bin_count_desc =  pd.crosstab(index=data_bins_desc[i],columns=data_bins_desc[y])
                                if 1 not in bin_count_desc.columns:
                                    bin_count_desc[1] = 0
                                if 0 not in bin_count_desc.columns:
                                    bin_count_desc[0] = 0
                                new_index=[]
                                for index in bin_count_desc.index:
                                        if index not in max_idx:
                                            new_index.append(index)
                                if len(new_index) ==0 :
                                    break
                                bin_count_new=bin_count_desc.loc[new_index,].copy()             
                                bin_count_cum = bin_count_new.cumsum()
                                bin_count_cum['good%']=bin_count_cum[0]/(bin_count_cum[0]+bin_count_cum[1])
                                max_idx1 = bin_count_cum.iloc[:,2].argmax()
                                if pd.isnull(max_idx1) :
                                    max_loc=0
                                else:
                                    max_loc = list(bin_count_cum.index).index(max_idx1)
                                if max_loc == 0 :
                                    max_idx[max_idx1]=max_idx1
                                    
                                else:#给出应合并的左右值
                                    a=bin_count_cum.index[0].left
                                    b=bin_count_cum.index[max_loc].right
                                    interval_new=pd.Interval(a,b)
                                    for x in  np.arange(max_loc+1):#对应更改明细数据分箱
                                        index_1 = bin_count_cum.index[x]
                                        data_bins_desc[i].where(data_bins_desc[i]!=index_1,interval_new,inplace=True)
                                    max_idx[interval_new]=interval_new
                            bin_count_fil = pd.crosstab(index=data_bins_desc[i],columns=data_bins_desc[y])
                            if 1 not in bin_count_fil.columns:
                                bin_count_fil[1]=0
                            if 0 not in bin_count_fil.columns:
                                bin_count_fil[0]=1
                            bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf') )
                            bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                            #判断是否有nan,并且nan的woe是否inf,若'是'则做以下处理：
                            data_nan=data_bins_desc.loc[:,[y,i]][pd.isnull(data_bins_desc[i])]
                            #if colname == 'call_close_2days_times_3m' :
                                #print('EEEEEEE',data_nan,'********8')
                            if data_nan.empty == False:
                                if len(data_nan[y].unique()) >1:
                                    data_nan['count']=1
                                    data_nan_cnt=data_nan.groupby(y).count()
                                else:
                                    if data_nan.iloc[0,0]==0:
                                        data_nan_cnt=pd.DataFrame([[0,list(data_nan.shape)[0]],[0,0]],index=[0.0, 1.0],columns=[m,'count'])
                                    else:
                                        data_nan_cnt=pd.DataFrame([[0,0],[0,list(data_nan.shape)[0]]],index=[0.0, 1.0],columns=[m,'count'])
                                #print('########################3',data_nan_cnt.columns)
                                woe_na = np.log((data_nan_cnt.loc[0,'count']/data_nan_cnt.loc[1,'count'])/(tot_good/tot_bad))
                                data_na_woe=pd.DataFrame([float('NaN')],index=[0],columns=['bin'])
                                data_na_woe[0]=data_nan_cnt.loc[0,'count']
                                data_na_woe[1]=data_nan_cnt.loc[1,'count']
                            else :
                                data_na_woe=pd.DataFrame([float('NaN')],index=[0],columns=['bin'])
                                data_na_woe[0]=0
                                data_na_woe[1]=0
                                woe_na =float('NaN')
                            if np.isinf(woe_na):#判断缺失值的Woe是否正负无穷
                                iv_na = 0
                                if (woe_na>0 and bin_count_fil.iloc[0,2]>bin_count_fil.iloc[len(bin_count_fil)-1,2]) or ( woe_na<0 and bin_count_fil.iloc[0,2]<bin_count_fil.iloc[len(bin_count_fil)-1,2]):
                                    bin_count_fil.iloc[0,0] = bin_count_fil.iloc[0,0]+data_nan_cnt.loc[0,'count']
                                    bin_count_fil.iloc[0,1] = bin_count_fil.iloc[0,1]+data_nan_cnt.loc[1,'count']
                                    bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf') )
                                    bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                                    woe_na=  bin_count_fil.iloc[0,2] 
                                    data_na_woe[0] = bin_count_fil.iloc[0,0] 
                                    data_na_woe[1] = bin_count_fil.iloc[0,1] 
                                    
                                else:
                                    bin_count_fil.iloc[len(bin_count_fil)-1,0] = bin_count_fil.iloc[len(bin_count_fil)-1,0]+data_nan_cnt.loc[0,'count']
                                    bin_count_fil.iloc[len(bin_count_fil)-1,1] = bin_count_fil.iloc[len(bin_count_fil)-1,1]+data_nan_cnt.loc[1,'count']
                                    bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf') )
                                    bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                                    woe_na=  bin_count_fil.iloc[len(bin_count_fil)-1,2] 
                                    data_na_woe[0] = bin_count_fil.iloc[len(bin_count_fil)-1,0] 
                                    data_na_woe[1] = bin_count_fil.iloc[len(bin_count_fil)-1,1] 
                            else:
                                iv_na = ((data_na_woe[0]/tot_good)-(data_na_woe[1]/tot_bad))*woe_na
        
                            data_na_woe['woe']=woe_na
                            data_na_woe['iv'] = iv_na 
                            for jj in np.arange(len(bin_count_fil)) :#判断分箱顶端是否足够量并且不为正负无穷
                                if (bin_count_fil.iloc[0,0]+bin_count_fil.iloc[0,1])>=(tot_good+tot_bad)*0.05  and np.isinf(bin_count_fil.iloc[0,2])==False :
                                    break
                                else:   
                                    if len(bin_count_fil)==1 :
                                        bin_count_fil.iloc[0,0]=tot_good
                                        bin_count_fil.iloc[0,1]=tot_bad
                                        bin_count_fil.iloc[0,2]=0
                                        data_na_woe['woe']=0
                                        data_na_woe['iv']=0
                                    else:
                                        if bin_count_fil.iloc[1,2] > data_na_woe['woe'].values and data_na_woe['woe'].values != bin_count_fil.iloc[0,2] :
                                            bin_count_fil.iloc[0,0]=bin_count_fil.iloc[0,0]+ data_na_woe[0].values
                                            bin_count_fil.iloc[0,1]=bin_count_fil.iloc[0,1] + data_na_woe[1].values
                                            bin_count_fil.iloc[0,2] = math.log(((bin_count_fil.iloc[0,0]/bin_count_fil.iloc[0,1])/(tot_good/tot_bad)))  if round(((bin_count_fil.iloc[0,0]/bin_count_fil.iloc[0,1])/(tot_good/tot_bad)),100)!=0 else float('-inf')
                                            data_na_woe['woe'] = bin_count_fil.iloc[0,2]
                                            data_na_woe['iv'] =0 
                                                
                                        else :
                                            left =bin_count_fil.index[0].left
                                            right = bin_count_fil.index[1].right
                                            interval_new2 = pd.Interval(left,right)
                                            index0 = bin_count_fil.index[0]
                                            index1 = bin_count_fil.index[1]
                                            data_bins_desc[i].where(data_bins_desc[i]!=index0,interval_new2,inplace=True)
                                            data_bins_desc[i].where(data_bins_desc[i]!=index1,interval_new2,inplace=True)
                                            bin_count_fil =  pd.crosstab(index=data_bins_desc[i],columns=data_bins_desc[y])
                                            if  1 not in bin_count_fil.columns:
                                                bin_count_fil[1]=0
                                            if 0 not in bin_count_fil.columns:
                                                bin_count_fil[0]=0
                                            bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf'))
                                            bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                                    

                            for jj in np.arange(len(bin_count_fil)) :#判断分箱底端是否足够量并且不为正负无穷
                                if (bin_count_fil.iloc[len(bin_count_fil)-1,0]+bin_count_fil.iloc[len(bin_count_fil)-1,1])>=(tot_good+tot_bad)*0.05  and np.isinf(bin_count_fil.iloc[len(bin_count_fil)-1,2])==False :
                                    break
                                else: 
                                    if len(bin_count_fil)==1 :
                                        bin_count_fil.iloc[0,0]=tot_good
                                        bin_count_fil.iloc[0,1]=tot_bad
                                        bin_count_fil.iloc[0,2]=0
                                        data_na_woe['woe']=0
                                        data_na_woe['iv']=0
                                    else:
                                        if bin_count_fil.iloc[len(bin_count_fil)-2,2]  < data_na_woe['woe'].values and data_na_woe['woe'].values != bin_count_fil.iloc[len(bin_count_fil)-1,2]  :
                                            bin_count_fil.iloc[len(bin_count_fil)-1,0]=bin_count_fil.iloc[len(bin_count_fil)-1,0]+ data_na_woe[0].values
                                            bin_count_fil.iloc[len(bin_count_fil)-1,1]=bin_count_fil.iloc[len(bin_count_fil)-1,1] + data_na_woe[1].values
                                            bin_count_fil.iloc[len(bin_count_fil)-1,2] = math.log(((bin_count_fil.iloc[len(bin_count_fil)-1,0]/bin_count_fil.iloc[len(bin_count_fil)-1,1])/(tot_good/tot_bad))) if round(((bin_count_fil.iloc[len(bin_count_fil)-1,0]/bin_count_fil.iloc[len(bin_count_fil)-1,1])/(tot_good/tot_bad)),100)!=0 else float('-inf')
                                            data_na_woe['woe'] = bin_count_fil.iloc[len(bin_count_fil)-1,2]
                                            data_na_woe['iv'] =0 
                                                
                                        else :
                                            left =bin_count_fil.index[len(bin_count_fil)-2].left
                                            right = bin_count_fil.index[len(bin_count_fil)-1].right
                                            interval_new2 = pd.Interval(left,right)
                                            index0 = bin_count_fil.index[len(bin_count_fil)-2]
                                            index1 = bin_count_fil.index[len(bin_count_fil)-1]
                                            data_bins_desc[i].where(data_bins_desc[i]!=index0,interval_new2,inplace=True)
                                            data_bins_desc[i].where(data_bins_desc[i]!=index1,interval_new2,inplace=True)
                                            bin_count_fil =  pd.crosstab(index=data_bins_desc[i],columns=data_bins_desc[y])
                                            if 1 not in bin_count_fil.columns:
                                                bin_count_fil[1]=0
                                            if 0 not in bin_count_fil.columns:
                                                bin_count_fil[0]=1
                                            bin_count_fil['woe'] =((bin_count_fil[0]/bin_count_fil[1])/(tot_good/tot_bad)).apply(lambda x:math.log(x)  if round(x,100) !=0 else float('-inf') )
                                            bin_count_fil['iv'] = (bin_count_fil[0]/tot_good - bin_count_fil[1]/tot_bad)*bin_count_fil['woe']
                            if len(bin_count_fil)<2 and  np.isinf(bin_count_fil['woe'].values) :
                                    bin_count_fil[0]=tot_good
                                    bin_count_fil[1]=tot_bad
                                    bin_count_fil['woe']=0
                                    bin_count_fil['iv'] =0
                                    data_na_woe[0]=tot_good
                                    data_na_woe[1]=tot_bad
                                    data_na_woe['woe']=0
                                    data_na_woe['iv']=0
                            #合并nan的woe iv
                            bin_count_fil['bin']=bin_count_fil.index
                            bin_count_fil=bin_count_fil.reindex(columns=['bin',0.0, 1.0, 'woe', 'iv'])
                            bin_count_final = pd.concat([bin_count_fil,data_na_woe],ignore_index=True)
                            tot_iv=bin_count_final['iv'].sum()
                            final_bins_desc.append(bin_count_final)
                            final_ivs_desc.append(tot_iv)
                        
                        #选出所有单调递增分箱尝试中最佳的一种分箱
                    max_iv_idx_desc = final_ivs_desc.index(max(final_ivs_desc))
                    final_bin_desc[colname]=final_bins_desc[max_iv_idx_desc]
                    final_iv_desc[colname]=max(final_ivs_desc)
                    data_bin_opt = data_bins_desc[max_iv_idx_desc+2]
                    for idx in final_bin_desc[colname]['bin']:  #将明细数据中的分箱替换成woe
                        data_bin_opt.where(data_bin_opt!=idx,final_bin_desc[colname][final_bin_desc[colname]['bin']==idx]['woe'].values,inplace=True)
                        #替换缺失值
                        na_w = final_bin_desc[colname][pd.isnull(final_bin_desc[colname]['bin'])]['woe'].values 
                        
                        if pd.isnull(na_w)==False:
                            data_bin_opt.where(pd.isnull(data_bin_opt)==False,na_w,inplace=True)
                        
                    if len(data_bin_desc)==0:
                        data_bin_desc['y']=data_bins_desc[y]
                    data_bin_desc[colname] =data_bin_opt
               
                #从递增递减中选取最优的一种分箱
                if  pearsonr(data_pearsonr.iloc[:,0],data_pearsonr.iloc[:,1])[0]>=0 :
                    final_iv[colname]=final_iv_desc[colname]
                    final_bin[colname]=final_bin_desc[colname]
                    data_bin[colname]=data_bin_desc[colname]
                else:
                    #print(final_iv_incr,'$$$$$$$$$$$$$$4',colname,'##############',final_iv_desc,'RRRRRRRRRR')
                    final_iv[colname]=final_iv_incr[colname]
                    final_bin[colname]=final_bin_incr[colname]
                    data_bin[colname]=data_bin_incr[colname]
                
        if data_charac.empty == False :
       
        #对所有分类变量分箱
            Data_new = pd.DataFrame( data[y])
    
    
            for colname in data_charac.columns:
                Data_new[colname] = data[colname]  
                Data_opt= Data_new[colname]
                Data_na = Data_new[pd.isnull(Data_new[colname])]
                if Data_na.empty == False:
                    if len(Data_na[y].unique())>1:
                        Data_na['count']=1
                        Data_na_cnt=Data_na.groupby(y).count()
                    else:
                        if Data_na.iloc[0,0]==0:
                            Data_na_cnt=pd.DataFrame([[0,list(Data_na.shape)[0]],[0,0]],index=[0.0, 1.0],columns=[colname,'count'])
                        else:
                            Data_na_cnt=pd.DataFrame([[0,0],[0,list(Data_na.shape)[0]]],index=[0.0, 1.0],columns=[colname,'count'])
                    Data_na_woe=pd.DataFrame([float('NaN')],index=[0],columns=['bin'])
                    Data_na_woe[0]=Data_na_cnt.loc[0,'count']
                    Data_na_woe[1]=Data_na_cnt.loc[1,'count']
                else :
                    Data_na_woe=pd.DataFrame([])
                Data_bin= pd.crosstab(index=data[colname],columns=data[y])
                if 1 not in Data_bin.columns:
                    Data_bin[1]=0
                if 0 not in Data_bin.columns:
                    Data_bin[0]=0
                Data_bin.where(pd.isnull(Data_bin)==False,0,inplace=True)
                Data_bin['bin'] = Data_bin.index
                Data_bin_index=Data_bin.reindex(columns=['bin',0.0, 1.0])
                Data_bins = pd.concat([Data_bin_index,Data_na_woe],ignore_index=True)
                Data_bins['woe']=((Data_bins[0]/Data_bins[1])/(tot_good/tot_bad)).apply(lambda x: math.log(x) if round(x,100) !=0 else float('-inf'))
                Data_bins['iv']=((Data_bins[0]/tot_good)-(Data_bins[1]/tot_bad))*Data_bins['woe']
                Data_bins=Data_bins.sort_values(by='woe')
                top_bins=[]
                bottom_bins=[]            
                for n in np.arange(len(Data_bins)):
                    
                    if Data_bins.iloc[0,1]+Data_bins.iloc[0,2]<(tot_good+tot_bad)*0.05 or np.isinf(Data_bins.iloc[0,3]):
                        Data_bins.iloc[1,1]=Data_bins.iloc[0,1]+Data_bins.iloc[1,1]
                        Data_bins.iloc[1,2]=Data_bins.iloc[0,2]+Data_bins.iloc[1,2]
                        top_drop = Data_bins.iloc[0,0]
                        Data_bins = Data_bins.iloc[1:,:].copy()
                        Data_bins['woe']=((Data_bins[0]/Data_bins[1])/(tot_good/tot_bad)).apply(lambda x: math.log(x) if round(x,100) !=0 else float('-inf'))
                        Data_bins['iv']=((Data_bins[0]/tot_good)-(Data_bins[1]/tot_bad))*Data_bins['woe']
                        top_bins.append(top_drop)
                    if Data_bins.iloc[len(Data_bins)-1,1]+Data_bins.iloc[len(Data_bins)-1,2]<(tot_good+tot_bad)*0.05 or np.isinf(Data_bins.iloc[len(Data_bins)-1,3]):
                        Data_bins.iloc[len(Data_bins)-1,1]=Data_bins.iloc[len(Data_bins)-1,1]+Data_bins.iloc[len(Data_bins)-2,1]
                        Data_bins.iloc[len(Data_bins)-2,2]=Data_bins.iloc[len(Data_bins)-1,2]+Data_bins.iloc[len(Data_bins)-2,2]
                        bottom_drop = Data_bins.iloc[len(Data_bins)-1,0]
                        Data_bins = Data_bins.iloc[:(len(Data_bins)-1),:].copy()
                        Data_bins['woe']=((Data_bins[0]/Data_bins[1])/(tot_good/tot_bad)).apply(lambda x: math.log(x) if round(x,100) !=0 else float('-inf'))
                        Data_bins['iv']=((Data_bins[0]/tot_good)-(Data_bins[1]/tot_bad))*Data_bins['woe']
                        bottom_bins.append(bottom_drop)
                top_woe = Data_bins.iloc[0,:]
                bottom_woe = Data_bins.iloc[len(Data_bins)-1,:]
                top_data =pd.DataFrame([])
                for bins in top_bins:
                    data_b = top_woe.copy()
                    data_b['bin']=bins
                    data_b['iv']=0
                    data_b2=pd.DataFrame(data_b).T
                    if top_data.empty :
                        top_data=data_b2
                    else:
                        top_data = pd.concat([top_data,data_b2],ignore_index=True)
                bottom_data =pd.DataFrame([])
                for bins in bottom_bins:
                    data_b = bottom_woe.copy()
                    data_b['bin']=bins
                    data_b['iv']=0
                    data_b2=pd.DataFrame(data_b).T
                    if bottom_data.empty :
                        bottom_data=data_b2
                    else:
                        bottom_data = pd.concat([bottom_data,data_b2],ignore_index=True) 
                Data_bins = pd.concat([Data_bins,bottom_data,top_data],ignore_index=True)
                        
                for cnt in np.arange(len(Data_bins)):
                        Data_opt.where(Data_opt!=Data_bins.iloc[cnt,0],Data_bins.iloc[cnt,3],inplace=True)  
                        na_w = Data_bins[pd.isnull(Data_bins.bin)]['woe'].values 
                        Data_opt.where(pd.isnull(Data_opt)==False,na_w,inplace=True)
        
                final_iv[colname] = Data_bins['iv'].sum()        
                final_bin[colname]=Data_bins
                data_bin[colname]=Data_opt

        for key in data_bin :
            data_bin[key] = data_bin[key].apply(lambda x :np.float16(x))
                
        self.woe_x=data_bin
        self.binning=final_bin
        self.iv=final_iv 
        self.y=y



    def Stepwise(self,min_iv=0.01,sle =0.025,sls=0.05):
        iv=self.iv
        response=self.y
        data=pd.DataFrame(self.woe_x)
        
        candidate_var = [response]
        for key in iv :
            if iv[key]>=min_iv :
                candidate_var.append(key)
                
        data_step= data[candidate_var].copy()
        
        remaining = set( data_step.columns)
        remaining.remove(response)
        selected=[]
        #print('############################3',remaining,'$$$$$$$$$$$','call_less_10s_cnt_all_in_percent' in remaining)
        while remaining:
           # while remaining:
            scores_with_candidates = []
            formula_0 = "{} ~ {} +1".format(response,'+'.join(selected))
            model_x_0 = smf.logit(formula_0,data_step).fit()
            r_square_0 = model_x_0.prsquared
            Multicol = []
            for candidate in remaining:
                #print(candidate,'$$$$')
                formula = "{} ~ {} +1".format(response,'+'.join(selected+[candidate]))
                try :
                    model_x = smf.logit(formula,data_step).fit()
                except LinAlgError  as e :
                    Multicol.append(candidate)
                    continue
                r_square = model_x.prsquared
                r_square_p = r_square - r_square_0
                p_value  = model_x.pvalues[candidate]
                coef_all =model_x.params
                coef = coef_all[candidate]
                if  (coef_all[1:]<0).all()  and p_value <=sle :
                    scores_with_candidates.append((r_square_p,p_value,coef,candidate))
            scores_with_candidates.sort()
            for var in Multicol :
                remaining.remove(var)
            if len(scores_with_candidates)>0 :  
                best_r_square_p, best_p_value,best_coef,best_candidate = scores_with_candidates.pop(-1) 
                #print(best_p_value,'@@@@',best_candidate)
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                #else:break
                
                formula_1 = "{}~{}+1".format(response,'+'.join(selected))
                model_x_1 =  smf.logit(formula_1,data_step).fit()
                p_value_1 =model_x_1.pvalues
                r_square_1 = model_x_1.prsquared
                worst_var =[]
                for var in selected :
                    waited = selected.copy()
                    waited.remove(var)
                    formula = "{}~{}+1".format(response,'+'.join(waited))
                    model_x =  smf.logit(formula,data_step).fit()
                    r_square_2 = model_x.prsquared
                    r_square_p = (r_square_1 -r_square_2)
                    worst_var.append((r_square_p ,p_value_1[var],var))
                worst_var.sort()
                worst_r_square,worst_p_value,worst_candidate = worst_var.pop(0)
            #print(worst_p_value,'@@@@',worst_candidate)

                if worst_p_value>sls:
                    selected.remove(worst_candidate)
                    remaining.add(worst_candidate)
                    if best_candidate == worst_candidate:break 
            else: break
        formula = "{}~{}+1".format(response,'+'.join(selected))
        model=smf.logit(formula,data_step).fit()
        #print(selected,'#######',remaining)
        self.model= model
        self.data=data_step
    
    
    
    def bin_to_score(self,P0=500,PDO=50,odds=2):
        
        model=self.model
        binning =self.binning
        
        selected_dict = dict(model.params)        
        selected_woe_dict={}
        
        for key in selected_dict :
            if key!='Intercept' :
                selected_woe_dict[key] = binning[key][['bin','woe']][binning[key]['woe']!=0]
                selected_woe_dict[key]['points']=(selected_dict['Intercept']/(len(selected_dict)-1)) +(selected_dict[key]*selected_woe_dict[key]['woe'])
    
    
    
        B = PDO/(math.log(2))
        A = P0+B*math.log(2)
        
        for key in selected_woe_dict :
            selected_woe_dict[key]['score'] = (A/(len(selected_dict)-1))-(B*selected_woe_dict[key]['points'])
         
        self.bin_score = selected_woe_dict
        

    
    # 计算AUC cut-off KS
    def compute_auc_ks(self,y_true=None,y_pred=None):
        if y_true is None or y_pred is None :
            data = self.data
            y = self.y
            y_true = data[y]
            y_pred = self.model.predict()
        fpr, tpr, thresholds = smt.roc_curve(y_true, y_pred, pos_label = 1)
        auc = smt.auc(fpr, tpr)
        imax = (tpr - fpr).argmax()
        ks = (tpr - fpr).max()
        cut_off = imax / len(tpr)
        #print('%7s  ' % ('[auc]'), '%f' % auc)
        #print('%7s  ' % ('[cut]'), '%f' % cut_off)
        #print('%7s  ' % ('[k-s]'), '%f' % ks)
        return auc ,ks ,cut_off

    
    # 绘制ROC曲线
    def plot_roc(self,y_true=None,y_pred=None):
        is_test=None
        if y_true is None or y_pred is None :
            data = self.data
            y = self.y
            y_true = data[y]
            y_pred = self.model.predict()
            is_test =1
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        fpr, tpr, thresholds = smt.roc_curve(y_true, y_pred, pos_label = 1)
        if is_test is None   :
            ax.set_title('roc curve:  test')
        else:
            ax.set_title('roc curve:  train')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid()
        ax.plot(fpr, tpr, 'r')
        ax.plot([0, 1], [0, 1], 'k--')
    
        
        
    # 绘制KS曲线
    def plot_ks(self,y_true=None,y_pred=None):
        
        is_test=None
        if y_true is None or y_pred is None :
            data = self.data
            y = self.y
            y_true = data[y]
            y_pred = self.model.predict()
            is_test=1
            

        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        fpr, tpr, thresholds = smt.roc_curve(y_true, y_pred, pos_label = 1)
        n_sample = len(tpr)
        x = [i / n_sample for i in range(n_sample)]
        imax = (tpr - fpr).argmax()
        cut_off = imax / len(tpr)
        max_tpr = tpr[imax]
        max_fpr = fpr[imax]
        if is_test is None :
            ax.set_title('K-S curve: test')
        else:
            ax.set_title('K-S curve: train')  
        ax.set_xlabel("Data Sets")
        ax.set_ylabel("Rate")
        ax.grid()
        ax.plot(x, tpr, 'r', label='True Positive Rate')
        ax.plot(x, fpr, 'b', label='False Positive Rate')
        ax.plot([cut_off, cut_off], [max_fpr, max_tpr], 'k--')
        ax.legend(loc="best")
    
    
    
    def  data_to_woe(self,data,oot=False ):
        
        model=self.model
        target=self.y
        binning=self.binning
        selected_variable = list(model.params.index.values)[1:]
        selected_variable.append(target)
        data =data[selected_variable].copy()

        
        for col in data.columns :
            if  col !=target :
                
                woe_frame = binning[col]
                col_id =list(data.columns).index(col)
                if data[col].dtype=='int64' or data[col].dtype=='float64' :
                    for i in np.arange(len(data[col])):
                        i_value = data.iloc[i,col_id]
                        for j in np.arange(len(woe_frame)):
                            interval = woe_frame.iloc[j,0]
                            if pd.isnull(interval) and pd.isnull(i_value) :
                                if pd.isnull(woe_frame.loc[j,'woe']):
                                    data.iloc[i,col_id] = min(woe_frame[pd.isnull(woe_frame['bin'])==False]['woe'])
                                else:
                                    data.iloc[i,col_id] = woe_frame.iloc[j,3]
                            if pd.isnull(interval)==False and i_value > interval.left and  i_value <= interval.right :
                                            data.iloc[i,col_id] = woe_frame.iloc[j,3]
                                            
                else:
                     for i in np.arange(len(data[col])):
                        i_value = data.iloc[i,col_id]
                        for j in np.arange(len(woe_frame)):
                            interval = woe_frame.iloc[j,0]
                            if pd.isnull(interval) and pd.isnull(i_value) :
                                data.iloc[i,col_id] = woe_frame.iloc[j,3]
                            if i_value == interval :
                                data.iloc[i,col_id] = woe_frame.iloc[j,3]              
                     for j in np.arange(len(data[col])):
                         if type(data.iloc[j,col_id])!=int and type(data.iloc[j,col_id])!=float :
                             data.iloc[j,col_id]= min(woe_frame['woe']) 
                                                    
        
        for col in data.columns :
           
            if   col!=target :
                data[col] = data[col].apply(lambda x:float(x))      
        if oot ==False :
            self.test_data = data 
        else :
            self.oot_data = data
        
        


    def PSI(self,oot = False ):#计算测试数据和训练数据之间的psi
        oot_data = self.oot_data
        test_data = self.test_data
        if oot == False :
            data = test_data
        else:
            data = oot_data
         
        y=self.y
        binning=self.binning
        psi={}
        selected_var =list(self.model.params.index.values)[1:]
        for key in selected_var:
            woe_data = binning[key]
            woe_data['total']=woe_data[0]+woe_data[1]
            woe_data['per'] = woe_data['total']/sum(woe_data['total'])
            woe_test = pd.DataFrame(data.groupby(key).count()[y])
            woe_test['per']=woe_test[y]/sum(woe_test[y])
            woe_test['woe']=woe_test.index.values
            psi_data = pd.merge(woe_data,woe_test,on='woe')
            psi_data['psi']=(psi_data['per_x'] - psi_data['per_y'] )*((psi_data['per_x']/psi_data['per_y']).apply(lambda x : math.log(x)  if round(x,100) !=0 else float('-inf') ))

            psi[key]=sum(psi_data['psi'])
        if oot ==False :  
            self.psi = psi 
        else:
            self.psi_oot = psi


                    