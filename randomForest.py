# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 20:02:41 2018

@author: Administrator
"""

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris

trees=10000
iris=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)
#n_estimators表示树的个数，测试中100颗树足够
forest=RandomForestClassifier(n_estimators=trees,random_state=0)
forest.fit(x_train,y_train)

print("random forest with %d trees:"%trees)  
print("accuracy on the training subset:{:.3f}".format(forest.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(forest.score(x_test,y_test)))
print('Feature importances:{}'.format(forest.feature_importances_))


n_features=iris.data.shape[1]
plt.barh(range(n_features),forest.feature_importances_,align='center')
plt.yticks(np.arange(n_features),iris.feature_names)
plt.title("random forest with %d trees:"%trees)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

'''
random forest with 10000 trees:
accuracy on the training subset:1.000
accuracy on the test subset:0.974
Feature importances:[ 0.1099358   0.02341364  0.45742098  0.40922957]
'''


