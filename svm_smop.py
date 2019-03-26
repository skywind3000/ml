#! /usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# svm_smo1.py - SVM Complete Platt SMO with kernel
#
# Created by skywind on 2019/03/24
# Last Modified: 2019/03/24 20:14:55
#
#======================================================================
from __future__ import print_function, unicode_literals
import sys
import random
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    for line in open(fileName):
        part = line.strip().split('\t')
        if not part:
            continue
        dataMat.append([float(part[0]), float(part[1])])
        labelMat.append(float(part[2]))
    return dataMat, labelMat


#----------------------------------------------------------------------
# compute K(xi . xj)
#----------------------------------------------------------------------
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Kernel not supported')
    return K


#----------------------------------------------------------------------
# opt struct
#----------------------------------------------------------------------
class optStruct:

    def __init__ (self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


#----------------------------------------------------------------------
# calculate E(k)
#----------------------------------------------------------------------
def calcEk(oS, k):
    fxk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fxk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    while True:
        j = int(random.uniform(0, m))
        if j != i:
            return j
    return -1

def clipAlpha(aj, H, L):
    return max(min(aj, H), L)

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = np.abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
    return True


#----------------------------------------------------------------------
# inner loop 
#----------------------------------------------------------------------
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or \
            (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - \
                oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - \
                oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - \
                oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - \
                oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) * 0.5
        return 1
    return 0


#----------------------------------------------------------------------
# SMO Platt
#----------------------------------------------------------------------
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


#----------------------------------------------------------------------
# test rbf
#----------------------------------------------------------------------
def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('data/testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet('data/testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
    return True




#----------------------------------------------------------------------
# 
#----------------------------------------------------------------------
def test1():
    dataArray, labelArray = loadDataSet("data/testSet.txt")
    b, alphas = smoP(dataArray, labelArray, 0.01, 0.001, 20000)
    print(b)
    print(alphas[alphas > 0])
    for i in range(100):
        if alphas[i] > 0:
            print(dataArray[i], labelArray[i])
    # print(alphas)
    return 0

def test2():
    testRbf()
    return 0


#----------------------------------------------------------------------
# testing suit 
#----------------------------------------------------------------------
if __name__ == '__main__':
    test2()

