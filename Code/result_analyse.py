#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 00:07:49 2019

@author: TH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
#from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv('analyse_exp/exp_data.csv')

data['Algo'].unique()
data2 = data.drop(th.index)
data = data2
dt = data[data['Algo'] == 'Decision tree']
knn = data[data['Algo'] == 'KNN']
svm = data[data['Algo'] == 'SVM']

dt.AUC

num_bin = 20

#plt.hist(dt.AUC, num_bin, facecolor='blue', alpha=0.5)
#plt.show()
#
#plt.hist(knn.AUC, num_bin, facecolor='blue', alpha=0.5)
#plt.show()
#
#plt.hist(svm.AUC, num_bin, facecolor='blue', alpha=0.5)
#plt.show()

auc_algo = [dt.AUC, knn.AUC, svm.AUC]
fig7, ax7 = plt.subplots()
ax7.set_title('Boxplot of the AUC of 3 algorithms')

ax7.boxplot(auc_algo, labels = ['Decision Tree', '5NN', 'SVM'])
#plt.xticks( ['Decision Tree', '5NN', 'SVM'])
plt.show()


#f1_algo = [dt.F1, knn.F1, svm.F1]
#fig8, ax8 = plt.subplots()
#ax8.set_title('Boxplot of the F1 of 3 algorithms')
#
#ax8.boxplot(f1_algo, labels = ['Decision Tree', '5NN', 'SVM'])
##plt.xticks( ['Decision Tree', '5NN', 'SVM'])
#plt.show()
stats.bartlett(dt.AUC, knn.AUC, svm.AUC)
f1, p1 = stats.kruskal(dt.AUC, knn.AUC, svm.AUC)
friedmanchisquare(dt.AUC, knn.AUC, svm.AUC)
#stats.levene(dt.AUC, knn.AUC, svm.AUC)

#mc = MultiComparison(data['Algo'], data['AUC'])

m_comp1 = pairwise_tukeyhsd(endog=data['AUC'], groups=data['Algo'], alpha=0.05)
#print(m_comp_res)

#mc_results = mc.tukeyhsd()
#print(mc_results)

print ("Stats = {}\np-value = {}\nTukey HSD: {}".format(f1, p1, m_comp1))
#%%

f1_algo = [dt.F1, knn.F1, svm.F1]
fig8, ax8 = plt.subplots()
ax8.set_title('Boxplot of the F1 of 3 algorithms')

ax8.boxplot(f1_algo, labels = ['Decision Tree', '5NN', 'SVM'])
plt.show()
#plt.xticks( ['Deci
stats.bartlett(dt.F1, knn.F1, svm.F1)
f2, p2 = stats.kruskal(dt.F1, knn.F1, svm.F1)
#stats.f_oneway
m_comp2 = pairwise_tukeyhsd(endog=data['F1'], groups=data['Algo'], alpha=0.05)
friedmanchisquare(dt.F1, knn.F1, svm.F1)
print ("Stats = {}\np-value = {}\nTukey HSD: {}".format(f2, p2, m_comp2))

#%%

data['resample'].unique()
ranos = data[data['resample'] == 'random oversampling']
smote = data[data['resample'] == 'SMOTE']
combine = data[(data['resample'] == 'SMOTE-ENN') | (data['resample'] == 'SMOTE-Tomeklinks')]
range_res = data[(data['resample'] == 'BorderlineSMOE1') | (data['resample'] == 'BorderlineSMOE2') |
        (data['resample'] == 'Safe Level SMOTE')]
clus = data[(data['resample'] == 'AHC') | (data['resample'] == 'CBSO') | (data['resample'] == 'MWMOTE')]
#data['resample']  'SMOTE-ENN' or 'SMOTE-Tomeklinks'

auc_resfam = [ranos.AUC, smote.AUC, combine.AUC, range_res.AUC, clus.AUC]
fig9, ax9 = plt.subplots()
ax9.set_title('Boxplot of the AUC of 5 Resample Families')
ax9.boxplot(auc_resfam, labels = ['RandomSampling', 'SMOTE', 'Combined', 'RangeRestrct'
                                  ,'ClusterBased'])
plt.show()
stats.bartlett(ranos.AUC, smote.AUC, combine.AUC, range_res.AUC, clus.AUC)
f3, p3 = stats.kruskal(ranos.AUC, smote.AUC, combine.AUC, range_res.AUC, clus.AUC)
print (f3, p3)

#m_comp3 = pairwise_tukeyhsd(endog=data['AUC'], groups=data['resample'], alpha=0.05)
#print (m_comp3)

#%%

auc_resfam = [ranos.F1, smote.F1, combine.F1, range_res.F1, clus.F1]
fig10, ax10 = plt.subplots()
ax10.set_title('Boxplot of the F1 of 5 Resample Families')
ax10.boxplot(auc_resfam, labels = ['RandomSampling', 'SMOTE', 'Combined', 'RangeRestrct'
                                  ,'ClusterBased'])
plt.show()
stats.bartlett(ranos.F1, smote.F1, combine.F1, range_res.F1, clus.F1)
f4, p4 = stats.kruskal(ranos.F1, smote.F1, combine.F1, range_res.F1, clus.F1)
print (f4, p4)

#%%

smoteenn = data[data['resample'] == 'SMOTE-ENN']
smotetomek = data[data['resample'] == 'SMOTE-Tomeklinks']

auc_rescomb = [ranos.AUC, smote.AUC, smoteenn.AUC, smotetomek.AUC]

fig11, ax11 = plt.subplots()
ax11.set_title('Boxplot of the AUC of the Combination Methods')
ax11.boxplot(auc_rescomb, labels = ['RandomSampling', 'SMOTE', 'SMOTE-ENN', 'SMOTE-Tomkelinks'])
plt.show()
stats.bartlett(ranos.AUC, smote.AUC, smoteenn.AUC, smotetomek.AUC)
f5, p5 = stats.kruskal(ranos.AUC, smote.AUC, smoteenn.AUC, smotetomek.AUC)
print (f5, p5)
#friedmanchisquare(ranos.AUC, smote.AUC, smoteenn.AUC, smotetomek.AUC)
np.mean(smoteenn.AUC)
np.mean(smotetomek.AUC)

#%%

f1_rescomb = [ranos.F1, smote.F1, smoteenn.F1, smotetomek.F1]

fig12, ax12 = plt.subplots()
ax12.set_title('Boxplot of the F1 of the Combination Methods')
ax12.boxplot(f1_rescomb, labels = ['RandomSampling', 'SMOTE', 'SMOTE-ENN', 'SMOTE-Tomkelinks'])
plt.show()
stats.bartlett(ranos.F1, smote.F1, smoteenn.F1, smotetomek.F1)
f6, p6 = stats.kruskal(ranos.F1, smote.F1, smoteenn.F1, smotetomek.F1)
#friedmanchisquare(ranos.F1, smote.F1, smoteenn.F1, smotetomek.F1)
print (f6, p6)

#%%

bdline1 = data[data['resample'] == 'BorderlineSMOE1']
bdline2 = data[data['resample'] == 'BorderlineSMOE2']
sfs = data[data['resample'] == 'Safe Level SMOTE']

auc_resres = [ranos.AUC, smote.AUC, bdline1.AUC, bdline2.AUC, sfs.AUC]

fig13, ax13 = plt.subplots()
ax13.set_title('Boxplot of the AUC of the Range Restricted Methods')
ax13.boxplot(auc_resres, labels = ['RandomSampling', 'SMOTE', 'Borderline1', 'Borderline2', 'SafeLevel'])
plt.show()
stats.bartlett(ranos.AUC, smote.AUC, bdline1.AUC, bdline2.AUC, sfs.AUC)
f7, p7 = stats.kruskal(ranos.AUC, smote.AUC, bdline1.AUC, bdline2.AUC, sfs.AUC)

print (f7, p7)

resres = [ranos, smote, bdline1, bdline2, sfs]
resres = pd.concat(resres)

m_comp4 = pairwise_tukeyhsd(endog=resres['AUC'], groups=resres['resample'], alpha=0.05)
print ("F = {}\np-value = {}\nTukey HSD:\n {}".format(f7, p7, m_comp4))

print ('M(SMOTE)={}, SD(SMOTE) = {}'.format(np.average(smote.AUC), np.std(smote.AUC)))
print ('M(SLS) = {}, SD(SLS) = {}'.format(np.average(sfs.AUC), np.std(sfs.AUC)))

#friedmanchisquare(ranos.AUC, smote.AUC, bdline1.AUC, bdline2.AUC, sfs.AUC)
#%%

f1_resres = [ranos.F1, smote.F1, bdline1.F1, bdline2.F1, sfs.F1]

fig14, ax14 = plt.subplots()
ax14.set_title('Boxplot of the F1 of the Range Restricted Methods')
ax14.boxplot(f1_resres, labels = ['RandomSampling', 'SMOTE', 'Borderline1', 'Borderline2', 'SafeLevel'])
plt.show()
stats.bartlett(ranos.F1, smote.F1, bdline1.F1, bdline2.F1, sfs.F1)
f8, p8 = stats.kruskal(ranos.F1, smote.F1, bdline1.F1, bdline2.F1, sfs.F1)
print (f8, p8)
#friedmanchisquare(ranos.F1, smote.F1, bdline1.F1, bdline2.F1, sfs.F1)
#%%

ahc = data[data['resample'] == 'AHC']
cbso = data[data['resample'] == 'CBSO']
mwmote = data[data['resample'] == 'MWMOTE']

auc_clus = [ranos.AUC, smote.AUC, ahc.AUC, cbso.AUC, mwmote.AUC]

fig15, ax15 = plt.subplots()
ax15.set_title('Boxplot of the AUC of the Clustering Based Methods')
ax15.boxplot(auc_clus, labels = ['RandomSampling', 'SMOTE', 'AHC', 'CBSO', 'MWMOTE'])
plt.show()
stats.bartlett(ranos.AUC, smote.AUC, ahc.AUC, cbso.AUC, mwmote.AUC)
f9, p9 = stats.kruskal(ranos.AUC, smote.AUC, ahc.AUC, cbso.AUC, mwmote.AUC)
print (f9, p9)
friedmanchisquare(ranos.AUC, smote.AUC, ahc.AUC, cbso.AUC, mwmote.AUC)

#%%

f1_clus = [ranos.F1, smote.F1, ahc.F1, cbso.F1, mwmote.F1]

fig16, ax16 = plt.subplots()
ax16.set_title('Boxplot of the F1 of the Clustering Based Methods')
ax16.boxplot(f1_clus, labels = ['RandomSampling', 'SMOTE', 'AHC', 'CBSO', 'MWMOTE'])
plt.show()
stats.bartlett(ranos.F1, smote.F1, ahc.F1, cbso.F1, mwmote.F1)
f10, p10 = stats.f_oneway(ranos.F1, smote.F1, ahc.F1, cbso.F1, mwmote.F1)
print (f10, p10)

#%%

file_ = open('Analysis By Dataset', 'a')


for i in data['data'].unique():
    t = data[data['data'] == i].sort_values(['AUC', 'F1'], ascending = [False, False]).head()[['data', 'resample','Algo', 'AUC', 'F1']]
    t.to_csv(r'DataANA.csv', header=None, index=None, sep=' ', mode='a')

file_.close()
#%%

import matplotlib.pyplot as plt

webpage_dt = data[(data['data'] == 'webpage') & (data['Algo'] == 'Decision tree')]
resample_ = (webpage_dt['resample'].unique())
y_pos = np.arange(len(resample_))

width = 0.4

fig11, ax11 = plt.subplots()



ax11.barh(y_pos, webpage_dt['AUC'],width, align='center', edgecolor='white', label='AUC')
ax11.barh(y_pos+width, webpage_dt['F1'],width, align='center',edgecolor='white', label='F1')
ax11.set_yticks(y_pos)
ax11.set_yticklabels(resample_)
ax11.invert_yaxis()  # labels read top-to-bottom
ax11.set_xlabel('Performance')
ax11.set_title('The Performances of Decision Tree on Webpage Dataset')
ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#%%

webpage_knn = data[(data['data'] == 'webpage') & (data['Algo'] == 'KNN')]
resample_ = (webpage_knn['resample'].unique())
y_pos = np.arange(len(resample_))

width = 0.4

fig11, ax11 = plt.subplots()



ax11.barh(y_pos, webpage_knn['AUC'],width, align='center', edgecolor='white', label='AUC')
ax11.barh(y_pos+width, webpage_knn['F1'],width, align='center',edgecolor='white', label='F1')
ax11.set_yticks(y_pos)
ax11.set_yticklabels(resample_)
ax11.invert_yaxis()  # labels read top-to-bottom
ax11.set_xlabel('Performance')
ax11.set_title('The Performances of k-NN on Webpage Dataset')
ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#%%

webpage_svm = data[(data['data'] == 'webpage') & (data['Algo'] == 'SVM')]
resample_ = (webpage_svm['resample'].unique())
y_pos = np.arange(len(resample_))

width = 0.4

fig11, ax11 = plt.subplots()



ax11.barh(y_pos, webpage_svm['AUC'],width, align='center', edgecolor='white', label='AUC')
ax11.barh(y_pos+width, webpage_svm['F1'],width, align='center',edgecolor='white', label='F1')
ax11.set_yticks(y_pos)
ax11.set_yticklabels(resample_)
ax11.invert_yaxis()  # labels read top-to-bottom
ax11.set_xlabel('Performance')
ax11.set_title('The Performances of SVM on Webpage Dataset')
ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#%%

#th = data[data['data'] == 'protein_homo']
#len(data)
#data2 = data.drop(th.index)

opt_dt = data[(data['data'] == 'optical_digits') & (data['Algo'] == 'Decision tree')]
resample_ = (opt_dt['resample'].unique())
y_pos = np.arange(len(resample_))

width = 0.4

fig12, ax12 = plt.subplots()

#plt.subplot(311)

ax12.barh(y_pos, opt_dt['AUC'],width, align='center', edgecolor='white', label='AUC')
ax12.barh(y_pos+width, opt_dt['F1'],width, align='center',edgecolor='white', label='F1')
ax12.set_yticks(y_pos)
ax12.set_yticklabels(resample_)
ax12.invert_yaxis()  # labels read top-to-bottom
ax12.set_xlabel('Performance')
ax12.set_title('The Performances of Decision Tree on Optical Digits Dataset')
ax12.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
#%%
opt_knn = data[(data['data'] == 'optical_digits') & (data['Algo'] == 'KNN')]
resample_ = (opt_knn['resample'].unique())
y_pos = np.arange(len(resample_))

width = 0.4

fig13, ax13 = plt.subplots()

#plt.subplot(312)

ax13.barh(y_pos, opt_knn['AUC'],width, align='center', edgecolor='white', label='AUC')
ax13.barh(y_pos+width, opt_knn['F1'],width, align='center',edgecolor='white', label='F1')
ax13.set_yticks(y_pos)
ax13.set_yticklabels(resample_)
ax13.invert_yaxis()  # labels read top-to-bottom
ax13.set_xlabel('Performance')
ax13.set_title('The Performances of kNN on Optical Digits Dataset')
ax13.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
#%%
opt_svm = data[(data['data'] == 'optical_digits') & (data['Algo'] == 'SVM')]
resample_ = (opt_svm['resample'].unique())
y_pos = np.arange(len(resample_))

width = 0.4

fig11, ax11 = plt.subplots()

#plt.subplot(313)

ax11.barh(y_pos, opt_svm['AUC'],width, align='center', edgecolor='white', label='AUC')
ax11.barh(y_pos+width, opt_svm['F1'],width, align='center',edgecolor='white', label='F1')
ax11.set_yticks(y_pos)
ax11.set_yticklabels(resample_)
ax11.invert_yaxis()  # labels read top-to-bottom
ax11.set_xlabel('Performance')
ax11.set_title('The Performances of SVM on Optical Digits Dataset')
ax11.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
#%%
def barplot_comparing(name, algo, dataset=data):
    vari = dataset[(dataset['data'] == name) & (dataset['Algo'] == algo)]
    resample_ = (vari['resample'].unique())
    y_ = np.arange(len(resample_))
    width = 0.4
    fig, ax = plt.subplots()
    ax.barh(y_, vari['AUC'], width, align='center', edgecolor='white',label='AUC')
    ax.barh(y_+width, vari['F1'], width, align='center', edgecolor='white',label='F1')
    ax.set_yticks(y_)
    ax.set_yticklabels(resample_)
    ax.invert_yaxis()
    ax.set_xlabel('Performance')
    ax.set_title('The Performance of {} on {}'.format(algo, name))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show
#%%
for i in data['data'].unique():
    barplot_comparing(i, 'Decision tree')