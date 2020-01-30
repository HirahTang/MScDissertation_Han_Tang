#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 00:36:39 2019

@author: TH
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:33:20 2019

@author: TH
"""
#from imblearn.datasets import fetch_datasets
import pandas as pd
import numpy as np
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
from sklearn.utils import Bunch
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


data_dict2 = {}
dt = pd.read_csv('dt/haberman.data', header = None)
#dt[dt[dt.columns[-1]] == 1][[3]] = -1
#dt[dt[dt.columns[-1]] == 2][[3]] = 1
mymap = {2:1, 1:-1}

dt[[3]] = dt[[3]].applymap(lambda s: mymap.get(s) if s in mymap else s)
haberman = Bunch(data = np.array(dt[dt.columns[0:-1]]), target = np.array(dt[dt.columns[-1]]))

data_dict2['haberman'] = haberman
#%%

dt2 = pd.read_csv('dt/glass.csv')
dt2
mymap = {5:1}
for i in range(1, 8):
    if i != 5:
        mymap[i] = -1
        
dt2[dt2.columns[-1]] = dt2[dt2.columns[-1]].astype(np.float32)
dt2[dt2.columns[-1]] = dt2[dt2.columns[-1]].map(lambda s: mymap.get(s) if s in mymap else s)
glass_5 = Bunch(data = np.array(dt2[dt2.columns[0:-1]]), target = np.array(dt2[dt2.columns[-1]]))

data_dict2['glass_5'] = glass_5

#main(data_dict2['glass_5'], 'glass_5')

#%%

glass = pd.read_csv('dt/glass.csv')
mymap_6 = {6:1}
for i in range(1, 8):
    if i != 6:
        mymap_6[i] = -1
glass_6 = glass
glass_6[glass_6.columns[-1]] = glass[glass.columns[-1]].map(lambda s: mymap_6.get(s) if s in mymap_6 else s)
glass_6 = Bunch(data = np.array(glass_6[glass_6.columns[0:-1]]), target = np.array(glass_6[glass_6.columns[-1]]))

data_dict2['glass_6'] = glass_6

#main(data_dict2['glass_6'], 'glass_6')
#%%

glass = pd.read_csv('dt/glass.csv')
mymap_7 = {7:1}
for i in range(1, 8):
    if i != 7:
        mymap_7[i] = -1
glass_7 = glass
glass_7[glass_7.columns[-1]] = glass[glass.columns[-1]].map(lambda s: mymap_7.get(s) if s in mymap_7 else s)
glass_7 = Bunch(data = np.array(glass_7[glass_7.columns[0:-1]]), target = np.array(glass_7[glass_7.columns[-1]]))

data_dict2['glass_7'] = glass_7

#main(data_dict2['glass_7'], 'glass_7')


#%%

heart = pd.read_csv('dt/heart.csv')

mymap_heart = {1:-1, 0:1}

heart[heart.columns[-1]] = heart[heart.columns[-1]].map(lambda s: mymap_heart.get(s) if s in mymap_heart else s)
heart = Bunch(data = np.array(heart[heart.columns[0:-1]]), target = np.array(heart[heart.columns[-1]]))
data_dict2['heart'] = heart



#main(data_dict2['heart'], 'heart')

#%%

ionosphere = pd.read_csv('dt/ionosphere_data_kaggle.csv')

mymap_iono = {'g':-1, 'b':1}

ionosphere[ionosphere.columns[-1]] = ionosphere[ionosphere.columns[-1]].map(lambda s: mymap_iono.get(s) if s in mymap_iono else s)
ionosphere = Bunch(data = np.array(ionosphere[ionosphere.columns[0:-1]]), target = np.array(ionosphere[ionosphere.columns[-1]]))
data_dict2['ionosphere'] = ionosphere
#main(data_dict2['ionosphere'], 'ionosphere')


#%%

#wine = pd.read_csv('dt/winequality-red.csv')
#
#wine.loc[wine.quality < 7] = -1
#wine.loc[wine.quality >= 7] = 1
#
#wine = Bunch(data = np.array(wine[wine.columns[0:-1]]), target = np.array(wine[wine.columns[-1]]))
#data_dict2['wine'] = wine
#main(data_dict2['wine'], wine)


#%%

pima = pd.read_csv('dt/pima-indians-diabetes.data.csv', header = None)
map_pima = {0:-1, 1:1}

pima[pima.columns[-1]] = pima[pima.columns[-1]].map(lambda s: map_pima.get(s) if s in map_pima else s)
pima = Bunch(data = np.array(pima[pima.columns[0:-1]]), target = np.array(pima[pima.columns[-1]]))
data_dict2['pima'] = pima
#main(data_dict2['pima'], 'pima')
#iris = load_iris()
#map_iris = {1:-1, 2:-1, 0:1}
#target2 = []
#for i in iris.target:
##    print (i)
#    if i == 1 or i == 2:
#        target2.append(-1)
#    else:
#        target2.append(1)
##print (target2)
#iris.target = target2
#data_dict2['iris_setosa'] = iris
#main(data_dict2['iris_setosa'], 'iris_setosa')
#%%

breastcc = pd.read_csv('dt/bcancerwisc.csv')
breastcc = breastcc.drop('id', axis = 1)
map_breast = {'M':1, 'B':-1}
breastcc[breastcc.columns[0]] = breastcc[breastcc.columns[0]].map(lambda s: map_breast.get(s) if s in map_breast else s)
breastcc = Bunch(data = np.array(breastcc[breastcc.columns[1:]]), target = np.array(breastcc[breastcc.columns[0]]))
data_dict2['Breast'] = breastcc
#main(data_dict2['Breast'], 'breast')

#%%

#forest = pd.read_csv('dt/covtype.csv')
#forestmap = {}
#for i in range(1, 8):
#    if i == 5 or i == 6 or i == 7:
#        forestmap[i] = 1
#    else:
#        forestmap[i] = -1
#        
#forest[forest.columns[-1]] = forest[forest.columns[-1]].map(lambda s: forestmap.get(s) if s in forestmap else s)
#forest = Bunch(data = np.array(forest[forest.columns[0:-1]]), target = np.array(forest[forest.columns[-1]]))
#
#data_dict2['forest'] = forest
#main(data_dict2['forest'], 'forest')
#%%
#dt = fetch_datasets()['ecoli']

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
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 2), random_state=1)
    svmclf = SVC(gamma='auto')
    
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
#                model = neigh.fit(X_resample, y_resample)
                model = svmclf.fit(X_resample, y_resample)
            
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
#            model = neigh.fit(X_resample, y_resample)
            model = svmclf.fit(X_resample, y_resample)
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
    result = open('OSTest_SVM_35_2','a')
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
    
#%%
    
f = open('OSTest_SVM_35_2','w+')
f.close()
for i in data_dict:  
    print (i)
    main(data_dict[i], i)
    
    
for i in data_dict2:
    print (i)
    main(data_dict2[i], i)
#    
#%%
#data_dict2['Breast'].data.shape    
#sum(data_dict2['Breast'].target == -1)/sum(data_dict2['Breast'].target == 1)
