#! /usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# LogisticRegression.py - 
#
# Created by skywind on 2019/03/15
# Last Modified: 2019/03/15 16:25:40
#
#======================================================================
from __future__ import print_function, unicode_literals
import sys
import time
import math
import random



#----------------------------------------------------------------------
# sigmoid
#----------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


#----------------------------------------------------------------------
# hypothesis
#----------------------------------------------------------------------
def hypothesis(x, w):
    return sigmoid(sum([ x[i] * w[i] for i, _ in enumerate(w) ]))


#----------------------------------------------------------------------
# incremental calculation
#----------------------------------------------------------------------
def LogisticStep(x, y, theta, alpha):
    cols = len(x)
    if cols == 0:
        raise ValueError('Feature size must great than zero.')
    h = hypothesis(x, theta)
    update = [ 1.0 ] * cols
    for j in range(cols):
        derivation = (h - y) * x[j]
        update[j] = theta[j] - alpha * derivation
    return update


#----------------------------------------------------------------------
# gradient descent
#----------------------------------------------------------------------
def LogisticRegression(x, y, theta, alpha, iterations, limit):
    rows = len(x)
    if rows == 0:
        raise ValueError('Data size must great than zero.')
    for count in range(iterations):
        for i in range(rows):
            theta = LogisticStep(x[i], y[i], theta, alpha)
        # calculate error
        error = 0
        for i in range(rows):
            h = hypothesis(x[i], theta)
            error += (h - y[i]) * (h - y[i])
        error = error / float(rows)
        if error < limit:
            break
    return theta


#----------------------------------------------------------------------
# LogisticRegression
#----------------------------------------------------------------------
def LogisticRegression2(x, y, theta, alpha, iterations, limit):
    rows = len(x)
    if rows == 0:
        raise ValueError('Data size must great than zero.')
    cols = len(x[0])
    theta = [ t for t in theta ]
    for count in range(iterations):
        # iterate feature columns
        update = [ 1.0 ] * cols
        for j in range(cols):
            # calculate derivation
            derivation = 0
            for i in range(rows):
                h = hypothesis(x[i], theta)
                derivation += (h - y[i]) * x[i][j]
            derivation = derivation
            # derivation = derivation
            update[j] = theta[j] - alpha * derivation
        # update theta
        theta = update
        # calculate error
        error = 0
        for i in range(rows):
            h = hypothesis(x[i], theta)
            error += (h - y[i]) * (h - y[i])
        error = error / float(rows)
        if error < limit:
            break
    return theta


#----------------------------------------------------------------------
# test
#----------------------------------------------------------------------
def TestLR(proc, x, y, alpha, maxiter, limit = 0.000001):
    k = [ random.random() for i, _ in enumerate(x[0]) ]
    t = time.time()
    k = proc(x, y, k, alpha, maxiter, limit)
    t = time.time() - t
    print('LogisticRegression time:', t)
    print('Theta:', k)
    X = [[-0.017612, 14.053064], [0.850433, 6.920334], [0.423363, 11.054677], [-2.168791, 0.143632]]
    Y = [0, 1, 0, 1]
    error = 0
    for i, _ in enumerate(X):
        h = hypothesis([1.0] + X[i], k)
        error += (h - Y[i]) * (h - Y[i])
        print(Y[i], h)
    error /= len(X)
    print('error', error)
    print()
    return 0



#----------------------------------------------------------------------
# testing suit 
#----------------------------------------------------------------------
input = '''
-0.017612   14.053064   0
-1.395634   4.662541    1
-0.752157   6.538620    0
-1.322371   7.152853    0
0.423363    11.054677   0
0.406704    7.067335    1
0.667394    12.741452   0
-2.460150   6.866805    1
0.569411    9.548755    0
-0.026632   10.427743   0
0.850433    6.920334    1
1.347183    13.175500   0
1.176813    3.167020    1
-1.781871   9.097953    0
-0.566606   5.749003    1
0.931635    1.589505    1
-0.024205   6.151823    1
-0.036453   2.690988    1
-0.196949   0.444165    1
1.014459    5.754399    1
1.985298    3.230619    1
-1.693453   -0.557540   1
-0.576525   11.778922   0
-0.346811   -1.678730   1
-2.124484   2.672471    1
1.217916    9.597015    0
-0.733928   9.098687    0
-3.642001   -1.618087   1
0.315985    3.523953    1
1.416614    9.619232    0
-0.386323   3.989286    1
0.556921    8.294984    1
1.224863    11.587360   0
-1.347803   -2.406051   1
1.196604    4.951851    1
0.275221    9.543647    0
0.470575    9.332488    0
-1.889567   9.542662    0
-1.527893   12.150579   0
-1.185247   11.309318   0
-0.445678   3.297303    1
1.042222    6.105155    1
-0.618787   10.320986   0
1.152083    0.548467    1
0.828534    2.676045    1
-1.237728   10.549033   0
-0.683565   -2.166125   1
0.229456    5.921938    1
-0.959885   11.555336   0
0.492911    10.993324   0
0.184992    8.721488    0
-0.355715   10.325976   0
-0.397822   8.058397    0
0.824839    13.730343   0
1.507278    5.027866    1
0.099671    6.835839    1
-0.344008   10.717485   0
1.785928    7.718645    1
-0.918801   11.560217   0
-0.364009   4.747300    1
-0.841722   4.119083    1
0.490426    1.960539    1
-0.007194   9.075792    0
0.356107    12.447863   0
0.342578    12.281162   0
-0.810823   -1.466018   1
2.530777    6.476801    1
1.296683    11.607559   0
0.475487    12.040035   0
-0.783277   11.009725   0
0.074798    11.023650   0
-1.337472   0.468339    1
-0.102781   13.763651   0
-0.147324   2.874846    1
0.518389    9.887035    0
1.015399    7.571882    0
-1.658086   -0.027255   1
1.319944    2.171228    1
2.056216    5.019981    1
-0.851633   4.375691    1
-1.510047   6.061992    0
-1.076637   -3.181888   1
1.821096    10.283990   0
3.010150    8.401766    1
-1.099458   1.688274    1
-0.834872   -1.733869   1
-0.846637   3.849075    1
1.400102    12.628781   0
1.752842    5.468166    1
0.078557    0.059736    1
0.089392    -0.715300   1
1.825662    12.693808   0
0.197445    9.744638    0
0.126117    0.922311    1
-0.679797   1.220530    1
0.677983    2.556666    1
0.761349    10.693862   0
-2.168791   0.143632    1
1.388610    9.341997    0
0.317029    14.739025   0
'''

X = []
Y = []


for line in input.split('\n'):
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) < 3:
        continue
    # print(parts)
    X.append((1.0, float(parts[0]), float(parts[1])))
    Y.append(float(parts[2]))


TestLR(LogisticRegression, X, Y, 0.01, 5000)
TestLR(LogisticRegression2, X, Y, 0.01, 5000)


