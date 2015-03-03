# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:39:11 2015

@author: sergulaydore
"""
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[0.],[.1],[.7],[1.],[1.1],[1.3],[1.4],[1.7],[2.1],[2.2]])
y = np.array([0.,0.,1.,0.,0.,0.,1.,1.,1.,1.])

clf = LogisticRegression(C=100000000)
clf.fit(X,y)
pred = clf.predict(X)

print clf.predict_log_proba(X)
print clf.predict_proba(X)
clf.get_params()
print clf.score(X,y)
print clf.predict(X)
clf.coef_
clf.intercept_