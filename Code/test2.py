#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:29:30 2019

@author: TH
"""

from imblearn.datasets import fetch_datasets
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, auc, accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
import smote_variants as sv
from imblearn.metrics import geometric_mean_score

# Fetch all test datasets in imblearn.datasets 

data_dict = {}
for i in fetch_datasets():
    data_dict[i] = fetch_datasets()[i]
    

ros = RandomOverSampler(random_state = 42)
smote = SMOTE(random_state = 42)
bdlsmote1 = BorderlineSMOTE(random_state = 42, kind = 'borderline-1')
bdlsmote2 = BorderlineSMOTE(random_state = 42, kind = 'borderline-2')
#adasyn = ADASYN(random_state = 42)
smoteenn = SMOTEENN(random_state = 42)
smotetomek = SMOTETomek(random_state = 42)
ahc = sv.AHC(random_state = 42)
slsmote = sv.Safe_Level_SMOTE(random_state = 42)
cbso = sv.CBSO(random_state = 42)
mwmote = sv.MWMOTE(random_state = 42)

res_1 = ['random_OS', 'SMOTE', 'Borderline1', 'Borderline2', 'SMOTE-ENN', 'SMOTE-Tomek']
#res_2 = ['Safe-Level-SMOTE', 'CBSO', 'MWMOTE']
res_dict = {'random_OS': ros,
            'SMOTE': smote,
            'Borderline1': bdlsmote1,
            'Borderline2': bdlsmote2,
            
            'SMOTE-ENN': smoteenn,
            'SMOTE-Tomek': smotetomek,
            'AHC': ahc,
            'Safe-Level-SMOTE': slsmote,
            'CBSO': cbso,
            'MWMOTE': mwmote}
result = {'random_OS': [],
          'SMOTE': [],
          'Borderline1': [],
          'Borderline2': [],
          
          'SMOTE-ENN': [],
          'SMOTE-Tomek': [],
          'AHC': [],
          'Safe-Level-SMOTE': [],
          'CBSO':[],
          'MWMOTE': []}        
def main(abalone, name, res = res_dict, res_result = result, name1 = res_1):
    
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    decision_tree = DecisionTreeClassifier(random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=5)
    
    for i in res:
        acc = []
        auc = []
        f1 = []
        gmean = []
#        precision = []
#        recall = []
#        print ("Resampling Method: {}".format(i))
# 10 Cross-Validation testing
        
        if abalone.data.shape[0] <= 10000:
        
            for train_index, test_index in cv.split(abalone.data):
                X_train, X_test, y_train, y_test = abalone.data[train_index], abalone.data[test_index], abalone.target[train_index], abalone.target[test_index]
                
                if i in name1:
                    X_resample, y_resample = res[i].fit_resample(X_train, y_train)
                else:
                    X_resample, y_resample = res[i].sample(X_train, y_train)
                model = neigh.fit(X_resample, y_resample)
            
                y_pred = model.predict(X_test)
                acc_unit, auc_unit, f1_unit, geo_unit = evaluate(y_test, y_pred)
# Measure metrices include Accuracy, AUC, F1, precision, recall
                acc.append(acc_unit)
                auc.append(auc_unit)
                f1.append(f1_unit)
                gmean.append(geo_unit)
#            precision.append(precision_unit)
#            recall.append(recall_unit)
        else:
            X_train, X_test, y_train, y_test = train_test_split(abalone.data, abalone.target, test_size=0.30, random_state=42)
            
            if i in name1:
                X_resample, y_resample = res[i].fit_resample(X_train, y_train)
            else:
                X_resample, y_resample = res[i].sample(X_train, y_train)
            model = neigh.fit(X_resample, y_resample)
            y_pred = model.predict(X_test)
            acc_unit, auc_unit, f1_unit, geo_unit = evaluate(y_test, y_pred)
# Measure metrices include Accuracy, AUC, F1, precision, recall
            acc.append(acc_unit)
            auc.append(auc_unit)
            f1.append(f1_unit)
            gmean.append(geo_unit)
            
            
        res_result[i] = [round(np.average(acc), 5), round(np.average(auc), 5), round(np.average(f1), 5), round(np.average(gmean), 5)]
    tableprint(res_result, name)    
        
def evaluate(test, pred):
    fpr, tpr, thresholds = roc_curve(test, pred)
#    print (confusion_matrix(test, pred))
    return accuracy_score(test, pred), float(auc(fpr, tpr)), f1_score(test, pred), geometric_mean_score(test, pred, average = 'macro')
#precision_score(test, pred), recall_score(test, pred)

def tableprint (r_dict, set_name):
    result = open('OSTest_KNN_5_27','a')
    result.write('\hline\n\multirow{4}{4em}{'+set_name+'}\n')
    result.write('& Acc & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\\n'.format(r_dict['random_OS'][0],r_dict['SMOTE'][0],
                 r_dict['Borderline1'][0],r_dict['Borderline2'][0]
                 ,r_dict['SMOTE-ENN'][0], 
           r_dict['SMOTE-Tomek'][0], r_dict['AHC'][0],r_dict['Safe-Level-SMOTE'][0],r_dict['CBSO'][0], 
           r_dict['MWMOTE'][0]))
    result.write('& AUC & {} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\\n'.format(r_dict['random_OS'][1],r_dict['SMOTE'][1],
                 r_dict['Borderline1'][1],r_dict['Borderline2'][1]
                 ,r_dict['SMOTE-ENN'][1], 
           r_dict['SMOTE-Tomek'][1], r_dict['AHC'][1], r_dict['Safe-Level-SMOTE'][1],r_dict['CBSO'][1], 
           r_dict['MWMOTE'][1]))
    result.write('& F1 & {} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\\n'.format(r_dict['random_OS'][2],r_dict['SMOTE'][2],
                 r_dict['Borderline1'][2],r_dict['Borderline2'][2]
                 ,r_dict['SMOTE-ENN'][2], 
           r_dict['SMOTE-Tomek'][2], r_dict['AHC'][2], r_dict['Safe-Level-SMOTE'][2],r_dict['CBSO'][2], 
           r_dict['MWMOTE'][2]))
    result.write('& G-Mean & {} & {} & {} & {} & {} & {} & {} & {} & {} & {}\\\\\n'.format(r_dict['random_OS'][3],r_dict['SMOTE'][3],
                 r_dict['Borderline1'][3],r_dict['Borderline2'][3]
                 ,r_dict['SMOTE-ENN'][3], 
           r_dict['SMOTE-Tomek'][3], r_dict['AHC'][3], r_dict['Safe-Level-SMOTE'][3],r_dict['CBSO'][3], 
           r_dict['MWMOTE'][3]))
    result.close()

#f = open('OSTest_KNN_5_27','w+')
#f.close()
#data2 = ['letter_img', 'ecoli', 'pen_digits', 'wine_quality', 'car_eval_4', 'mammography', 'abalone_19']
#for i in data2:
#    print (i)
#    main(data_dict[i], i)
#main(data_dict['protein_homo'], 'protein_homo')


#main(data_dict['ecoli'], 'ecoli')
#for i in data_dict:
#    print (i)
#    main(data_dict[i], i)
#X_resample, y_resample = res_dict['CBSO'].sample(data_dict['abalone'].data, data_dict['abalone'].target)

#%%
#adult = pd.read_csv('data/adult.csv')
#
#
#
#
#adult.data = adult[adult.columns[0:-1]]
#adult.target = adult[adult.columns[-1]]
#
#main(adult, 'adult')


data_dict
