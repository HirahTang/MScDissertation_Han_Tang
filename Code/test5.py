#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:13:35 2019

@author: TH
"""
#from sklearn.preprocessing import OneHotEncoder
from imblearn.datasets import fetch_datasets
import pandas as pd
import numpy as np
from sklearn.utils import Bunch 
from sklearn.datasets import load_iris

#enc = OneHotEncoder(handle_unknown='ignore')

data_dict2 = {}
dt = pd.read_csv('dt/haberman.data', header = None)
#dt[dt[dt.columns[-1]] == 1][[3]] = -1
#dt[dt[dt.columns[-1]] == 2][[3]] = 1
mymap = {2:1, 1:-1}

dt[[3]] = dt[[3]].applymap(lambda s: mymap.get(s) if s in mymap else s)
haberman = Bunch(data = np.array(dt[dt.columns[0:-1]]), target = np.array(dt[dt.columns[-1]]))
data_dict2['haberman'] = haberman

#%%

glass = pd.read_csv('dt/glass.csv')
#dt2
mymap_5 = {5:1}
for i in range(1, 8):
    if i != 5:
        mymap_5[i] = -1
        
glass[glass.columns[-1]] = glass[glass.columns[-1]].astype(np.float32)
glass_5 = glass
glass_5[glass_5.columns[-1]] = glass[glass.columns[-1]].map(lambda s: mymap_5.get(s) if s in mymap_5 else s)
glass_5 = Bunch(data = np.array(glass_5[glass_5.columns[0:-1]]), target = np.array(glass_5[glass_5.columns[-1]]))

#%%
glass = pd.read_csv('dt/glass.csv')
mymap_6 = {6:1}
for i in range(1, 8):
    if i != 6:
        mymap_6[i] = -1
glass_6 = glass
glass_6[glass_6.columns[-1]] = glass[glass.columns[-1]].map(lambda s: mymap_6.get(s) if s in mymap_6 else s)
glass_6 = Bunch(data = np.array(glass_6[glass_6.columns[0:-1]]), target = np.array(glass_6[glass_6.columns[-1]]))
#%%

glass = pd.read_csv('dt/glass.csv')
mymap_7 = {7:1}
for i in range(1, 8):
    if i != 7:
        mymap_7[i] = -1
glass_7 = glass
glass_7[glass_7.columns[-1]] = glass[glass.columns[-1]].map(lambda s: mymap_7.get(s) if s in mymap_7 else s)
glass_7 = Bunch(data = np.array(glass_7[glass_7.columns[0:-1]]), target = np.array(glass_7[glass_7.columns[-1]]))

#%%

heart = pd.read_csv('dt/heart.csv')

mymap_heart = {1:-1, 0:1}

heart[heart.columns[-1]] = heart[heart.columns[-1]].map(lambda s: mymap_heart.get(s) if s in mymap_heart else s)
heart = Bunch(data = np.array(heart[heart.columns[0:-1]]), target = np.array(heart[heart.columns[-1]]))
data_dict2['heart'] = heart


#%%
ionosphere = pd.read_csv('dt/ionosphere_data_kaggle.csv')

mymap_iono = {'g':-1, 'b':1}

ionosphere[ionosphere.columns[-1]] = ionosphere[ionosphere.columns[-1]].map(lambda s: mymap_iono.get(s) if s in mymap_iono else s)
ionosphere = Bunch(data = np.array(ionosphere[ionosphere.columns[0:-1]]), target = np.array(ionosphere[ionosphere.columns[-1]]))
data_dict2['ionosphere'] = ionosphere
#%%
#iris = pd.read_csv('dt/Iris.csv')
#iris = iris.drop('Id', axis = 1)
#map_iris_set = {}
#for i in iris[iris.columns[-1]].unique():
#    if i == 'Iris-setosa':
#        map_iris_set[i] = 1
#    else:
#        map_iris_set[i] = -1
#        
#iris[iris.columns[-1]] = iris[iris.columns[-1]].map(lambda s: map_iris_set.get(s) if s in map_iris_set else s)
#iris_1 = Bunch(data = np.array(iris[iris.columns[0:-1]]), target = np.array(iris[iris.columns[-1]]))
#data_dict2['iris_setosa'] = iris_1
#%%

iris = load_iris()
map_iris = {1:-1, 2:-1, 0:1}
target2 = []
for i in iris.target:
#    print (i)
    if i == 1 or i == 2:
        target2.append(-1)
    else:
        target2.append(1)
#print (target2)
iris.target = target2

#%%

pima = pd.read_csv('dt/pima-indians-diabetes.data.csv', header = None)
map_pima = {0:-1, 1:1}

pima[pima.columns[-1]] = pima[pima.columns[-1]].map(lambda s: map_pima.get(s) if s in map_pima else s)
pima = Bunch(data = np.array(pima[pima.columns[0:-1]]), target = np.array(pima[pima.columns[-1]]))
data_dict2['pima'] = pima

#%%

wine = pd.read_csv('dt/winequality-red.csv')

wine.loc[wine.quality < 7] = -1
wine.loc[wine.quality >= 7] = 1

wine = Bunch(data = np.array(wine[wine.columns[0:-1]]), target = np.array(wine[wine.columns[-1]]))
data_dict2['wine'] = wine

#iris.target = iris.target.applymap(lambda s: map_iris.get(s) if s in map_iris else s)
#enc
#ec = Bunch(data = np.array(ec.data), target = np.array(ec.target))
##ad.data
#
#ecec
#ecec2 = fetch_datasets()['ecoli']

#%%

breastcc = pd.read_csv('dt/bcancerwisc.csv')
breastcc = breastcc.drop('id', axis = 1)
map_breast = {'M':1, 'B':-1}
breastcc[breastcc.columns[0]] = breastcc[breastcc.columns[0]].map(lambda s: map_breast.get(s) if s in map_breast else s)
breastcc = Bunch(data = np.array(breastcc[breastcc.columns[1:]]), target = np.array(breastcc[breastcc.columns[0]]))

#%%

forest = pd.read_csv('dt/covtype.csv')
forestmap = {}
for i in range(1, 8):
    if i == 5 or i == 6 or i == 7:
        forestmap[i] = 1
    else:
        forestmap[i] = -1
        
forest[forest.columns[-1]] = forest[forest.columns[-1]].map(lambda s: forestmap.get(s) if s in forestmap else s)
forest = Bunch(data = np.array(forest[forest.columns[0:-1]]), target = np.array(forest[forest.columns[-1]]))

#%%

liver = pd.read_csv('dt/indian_liver_patient.csv')

#%%

data_dict = {}
for i in fetch_datasets():
    data_dict[i] = fetch_datasets()[i]
    
#%%
    
pen = data_dict['pen_digits'].data
