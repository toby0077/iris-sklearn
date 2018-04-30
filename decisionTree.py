# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 20:12:51 2018

@author: Administrator
"""

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


iris=load_iris()
names=iris.feature_names
X_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)
#调参
list_average_accuracy=[]
depth=range(1,30)
for i in depth:
    #max_depth=4限制决策树深度可以降低算法复杂度，获取更精确值
    tree= DecisionTreeClassifier(max_depth=i,random_state=0)
    tree.fit(X_train,y_train)
    accuracy_training=tree.score(X_train,y_train)
    accuracy_test=tree.score(x_test,y_test)
    average_accuracy=(accuracy_training+accuracy_test)/2.0
    #print("average_accuracy:",average_accuracy)
    list_average_accuracy.append(average_accuracy)
    
max_value=max(list_average_accuracy)
#索引是0开头，结果要加1
best_depth=list_average_accuracy.index(max_value)+1
print("best_depth:",best_depth)

best_tree= DecisionTreeClassifier(max_depth=best_depth,random_state=0)
best_tree.fit(X_train,y_train)
accuracy_training=best_tree.score(X_train,y_train)
accuracy_test=best_tree.score(x_test,y_test)

print("decision tree:")    
print("accuracy on the training subset:{:.3f}".format(best_tree.score(X_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(best_tree.score(x_test,y_test)))

'''
best_depth: 4
decision tree:
accuracy on the training subset:1.000
accuracy on the test subset:0.974
'''
#生成一个dot文件，以后用cmd形式生成图片
export_graphviz(best_tree,out_file="Iris.dot",class_names=['setosa','versicolor','virginica'],feature_names=names,impurity=False,filled=True)



