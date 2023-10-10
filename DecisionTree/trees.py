#! /usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# trees.py - 
#
# Created by skywind on 2018/07/06
# Last Modified: 2018/07/06 22:12:37
#
#======================================================================
import math
import sys

# entropy
def calcShannonEnt(data):
    numEntities = len(data)
    labelCounts = {}
    for featVec in data:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntities
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [ [1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] + featVec[axis + 1:]
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # print 'baseEntropy', baseEntropy
    for i in range(numFeatures):
        featList = [ example[i] for example in dataSet ]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # print 'index', i, 'entropy', '%.4f'%newEntropy, 'infoGain', infoGain
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature



#----------------------------------------------------------------------
# testing case
#----------------------------------------------------------------------
if __name__ == '__main__':
    def test1():
        dataSet, labels = createDataSet()
        index = chooseBestFeatureToSplit(dataSet)
        print('best feature', index)
        return 0
    test1()




