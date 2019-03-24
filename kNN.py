from __future__ import print_function
import numpy as np
import operator
import os
import sys

if sys.version_info[0] >= 3:
    xrange = range

def createDataSet():
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# kNN classifier
def classify0(inX, dataSet, labels, k):
    # calculate distance
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # calculate minimal distance
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
            key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


# load image
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fp = open(filename)
    for i in xrange(32):
        lineStr = fp.readline()
        for j in xrange(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# hand writing classifier
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('data/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('data/digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('data/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is %d\n"%(
            classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print('the total number of error is: %d' % errorCount)
    print('the total error rate is: %f'%(errorCount / float(mTest)))
    return 0

# testing case
if __name__ == '__main__':
    def test1():
        group, labels = createDataSet()
        print(classify0([0,0], group, labels, 3))
        return 0
    def test2():
        testVector = img2vector('data/digits/testDigits/0_13.txt')
        print(testVector[0,0:31])
    def test3():
        handwritingClassTest()
    test3()

