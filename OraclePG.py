#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:43:17 2019

@author: quentin
"""
import numpy as np
from Structures_N import *
from Probleme_R import *


def F(qc):
    x1 = q0 + np.dot(B,qc)
    x2 = r*x1*np.abs(x1)
    res = 1/3*(np.dot(x1,x2)) + np.dot(pr,np.dot(Ar,x1))
    return res

def G(qc):
    x = q0+np.dot(B,qc)
    res = np.dot(np.transpose(np.dot(Ar,B)),pr) + np.dot(np.transpose(B),r*np.abs(x)*x)
    return res

def H(qc):
    res = 2*np.dot(np.transpose(B),np.dot(np.diag(r*np.abs(q0+np.dot(B,qc))),B))
    return res

def OraclePG(qc,ind=4):
    if ind==2:
        return F(qc)
    if ind==3:
        return G(qc)
    if ind==4:
        return F(qc),G(qc)
    return


def OraclePH(qc,ind=7):
    if ind==2:
        return F(qc)
    if ind==3:
        return G(qc)
    if ind==4:
        return F(qc),G(qc)
    if ind==5:
        return H(qc)
    if ind==6:
        return G(qc),H(qc)
    if ind==7:
        return F(qc),G(qc),H(qc)
    return
