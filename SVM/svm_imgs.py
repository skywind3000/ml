#! /usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# svm_imgs.py - 
#
# Created by skywind on 2019/03/24
# Last Modified: 2019/03/24 23:53:33
#
#======================================================================
from __future__ import print_function, unicode_literals
import sys
import time
import os
import numpy as np
import svm_smop


# load image
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fp = open(filename)
    for i in range(32):
        lineStr = fp.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    hwLabels = []
    trainingFileList = os.listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('../data/svm-digits/trainingDigits')
    print('fuck')
    b, alphas = svm_smop.smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    print('suck')
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = svm_smop.kernelTrans(sVs, datMat[i,:], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): 
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages("../data/svm-digits/testDigits")
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = svm_smop.kernelTrans(sVs, datMat[i,:], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
    return True



#----------------------------------------------------------------------
# testing suit 
#----------------------------------------------------------------------
if __name__ == '__main__':
    def test1():
        testDigits()
        return 0
    test1()



