# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 08:09:03 2021

@author: Felix
"""


import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

X = np.loadtxt('X.csv', delimiter =',');
y = np.loadtxt('y.csv', delimiter = ',');


def compute_alpha(X, y, ridge = False, label = ""):
    s = [.1,.3,.7,.9,1.1,1.3,1.5]
    alphas = []
    if(ridge):
        for i in range(len(s)):
            alphas.append(LA.inv(X.T @ X + s[i] * np.identity(50)) @ X.T @ y.T)
        return alphas
    else:
         alpha = LA.inv(X.T @ X) @ X.T @ y.T
         return alpha
     
def compute_error(X,y, alpha_s, ridge = False, label = ""):
    s = [.1,.3,.7,.9,1.1,1.3,1.5]
    if(ridge):
        errors = []
        print(f'\nError Values For RR Alphas {label}\n')
        for i in range(len(alpha_s)):
            errors.append(LA.norm(y - np.dot(X,alpha_s[i]))**2)
            print(f's: {s[i]}\tValue: {round(errors[i],3)}')
        print()
        return errors
    else:
        print(f'Error value for OLE Alpha {label}:\n')   
        alpha_error = (LA.norm(y - np.dot(X,alpha_s)))**2
        print(alpha_error)
        return alpha_error
    

alpha_ole_full = compute_alpha(X,y,ridge = False, label = "Full Data")
ole_full_error = compute_error(X,y, alpha_ole_full, ridge = False, label = "Full Data")

alpha_rr_full = compute_alpha(X,y,ridge = True, label = "Full Data")
rr_full_error = compute_error(X,y, alpha_rr_full, ridge = True, label = "Full Data")

print("-------------------------------------------------------")

           
X1 = X[:75,:]
Y1 = y[:75]

X1_test = X[75:,:]
Y1_test = y[75:]

X2 = X[25:,:]
Y2 = y[25:]

X2_test = X[:25,:]
Y2_test = y[:25]


X3 = np.vstack((X[:50,:], X[75:,:]))
Y3 = np.concatenate((y[:50], y[75:]))

X3_test = np.vstack((X[50:,:], X[:75,:]))
Y3_test = np.concatenate((y[50:], y[:75]))

X4 = np.vstack((X[:25,:], X[50:,:]))
Y4 = np.concatenate((y[:25], y[50:]))

X4_test = np.vstack((X[25:,:], X[:50,:]))
Y4_test = np.concatenate((y[25:], y[:50]))

ole_errors= []
rr_errors = []

alpha_ole_xyOne = compute_alpha(X1,Y1,ridge = False, label = "xyOne")
ole_xyOne_error = compute_error(X1_test,Y1_test, alpha_ole_full, ridge = False, label = "xyOne")


alpha_rr_xyOne = compute_alpha(X1,Y1,ridge = True, label = "xyOne")
rr_xyOne_error = compute_error(X1_test,Y1_test, alpha_rr_full, ridge = True, label = "xyOne")

print("-------------------------------------------------------")

alpha_ole_xyTwo = compute_alpha(X2,Y2,ridge = False, label = "xyTwo")
ole_xyTwo_error = compute_error(X2_test,Y2_test, alpha_ole_full, ridge = False, label = "xyTwo")


alpha_rr_xyTwo = compute_alpha(X2,Y2,ridge = True, label = "xyTwo")
rr_xyTwo_error = compute_error(X2_test,Y2_test, alpha_rr_full, ridge = True, label = "xyTwo")

print("-------------------------------------------------------")

alpha_ole_xyThree = compute_alpha(X3,Y3,ridge = False, label = "xyThree")
ole_xyThree_error = compute_error(X3_test,Y3_test, alpha_ole_full, ridge = False, label = "xyThree")


alpha_rr_xyThree = compute_alpha(X3,Y3,ridge = True, label = "xyThree")
rr_xyThree_error = compute_error(X3_test,Y3_test, alpha_rr_full, ridge = True, label = "xyThree")

print("-------------------------------------------------------")

alpha_ole_xyFour = compute_alpha(X4,Y4,ridge = False, label = "xyFour")
ole_xyFour_error = compute_error(X4,Y4, alpha_ole_full, ridge = False, label = "xyFour")


alpha_rr_xyFour = compute_alpha(X4,Y4,ridge = True, label = "xyFour")
rr_xyFour_error = compute_error(X4_test,Y4_test, alpha_rr_full, ridge = True, label = "xyFour")

cross_ = {}
s = [.1,.3,.7,.9,1.1,1.3,1.5]

print("-------------------------------------------------------")

subset_errors = []
