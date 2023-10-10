#! /usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# svm_smo1.py - SVM SMO simple
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

def selectJrand(i, m):
    while True:
        j = int(random.uniform(0, m))
        if j != i:
            return j
    return -1

def clipAlpha(aj, H, L):
    return max(min(aj, H), L)


# 简单 SMO 求解系数 alpha
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 计算 f(xi) = sum[ (a * y) * (xj . xiT) ] + b
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            # 计算误差 Ei = f(xi) - y[i]
            Ei = fXi - float(labelMat[i])
            # 检测是否违法 KKT 约束
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or \
                    (labelMat[i] * Ei > toler and alphas[i] > 0):
                j = selectJrand(i, m)
                # 计算 f(xj)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                # 计算 f(xj) 的误差
                Ej = fXj - float(labelMat[j])
                # 保存系数
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证处于区间范围
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if H == L:
                    print("H == L")
                    continue
                # eta = 2 * (xi . xjT) - (xi . xiT)
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
                    dataMatrix[i,:] * dataMatrix[i,:].T - \
                    dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if np.abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 以 alphas[j] 的增量更新 alphas[i]，但是方向相反，保证 KKT 条件
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - \
                    labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - \
                    labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) * 0.5
                alphaPairsChanged += 1
                print("Iteration: %d i:%d, pairs changed %s" % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas


#----------------------------------------------------------------------
# 
#----------------------------------------------------------------------
def test1():
    dataArray, labelArray = loadDataSet("../data/testSet.txt")
    b, alphas = smoSimple(dataArray, labelArray, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas > 0])
    for i in range(100):
        if alphas[i] > 0:
            print(dataArray[i], labelArray[i])
    # print(alphas)



#----------------------------------------------------------------------
# testing suit 
#----------------------------------------------------------------------
if __name__ == '__main__':
    test1()

