#! /usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# GradientDescent.py - 
#
# Created by skywind on 2019/03/14
# Last Modified: 2019/03/14 21:00:27
#
#======================================================================
from __future__ import print_function, unicode_literals
import sys
import time


#----------------------------------------------------------------------
# GradientDescent
#----------------------------------------------------------------------
def GradientDescent(x, y, theta, alpha, iterations, limit):
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
                h = sum([ x[i][n] * theta[n] for n in range(cols) ])
                derivation += (h - y[i]) * x[i][j]
            derivation = derivation / float(rows)
            update[j] = theta[j] - alpha * derivation
        # update theta
        theta = update
        # calculate error
        error = 0
        for i in range(rows):
            h = sum([ x[i][n] * theta[n] for n in range(cols) ])
            error += (h - y[i]) * (h - y[i])
        error = error / float(rows)
        if error < limit:
            break
    return theta


#----------------------------------------------------------------------
# GradientDescent
#----------------------------------------------------------------------
def GradientStep(x, y, theta, alpha):
    cols = len(x)
    if cols == 0:
        raise ValueError('Feature size must great than zero.')
    h = sum([ x[n] * theta[n] for n in range(cols) ])
    update = [ 1.0 ] * cols
    for j in range(cols):
        derivation = (h - y) * x[j]
        update[j] = theta[j] - alpha * derivation
    return update


#----------------------------------------------------------------------
# Increamental GradientDescent
#----------------------------------------------------------------------
def GradientDescent2(x, y, theta, alpha, iterations, limit):
    rows = len(x)
    if rows == 0:
        raise ValueError('Data size must great than zero.')
    cols = len(x[0])
    for count in range(iterations):
        for i in range(rows):
            theta = GradientStep(x[i], y[i], theta, alpha)
        # calculate error
        error = 0
        for i in range(rows):
            h = sum([ x[i][n] * theta[n] for n in range(cols) ])
            error += (h - y[i]) * (h - y[i])
        error = error / float(rows)
        if error < limit:
            break
    return theta


#----------------------------------------------------------------------
# numpy
#----------------------------------------------------------------------
def GradientDescent3(x, y, theta, alpha, iterations, limit):
    rows = len(x)
    if rows == 0:
        raise ValueError('Data size must great than zero.')
    import numpy
    x = numpy.array(x)
    y = numpy.array(y)
    theta = numpy.array([ t for t in theta ])
    t = x.transpose()
    for count in range(iterations):
        h = numpy.dot(x, theta)
        error = h - y
        derivation = numpy.dot(t, error) / float(rows)
        theta = theta - float(alpha) * derivation
        # update error
        e = numpy.dot(x, theta) - y
        s = sum(e * e) / float(rows)
        if s < limit:
            break
    return theta


#----------------------------------------------------------------------
# test 
#----------------------------------------------------------------------
def TestGD(proc, x, y, alpha, maxiter, limit = 0.0001):
    k = [ 1 ] * len(x[0])
    t = time.time()
    k = proc(x, y, k, alpha, maxiter, limit)
    t = time.time() - t
    print('GradientDescent time:', t)
    print('Theta:', k)
    error = 0
    X = [[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]]
    Y = [9.5, 10.2, 10.9, 11.6, 12.3]
    for i in range(len(X)):
        h = sum([ X[i][n] * k[n] for n in range(len(k)) ])
        error += (h - Y[i]) * (h - Y[i])
        print(Y[i], h)
    error /= len(X)
    print('error', error)
    print()
    return 0



#----------------------------------------------------------------------
# Samples
#----------------------------------------------------------------------
X = [[1.1, 1.5], [1.3, 1.9], [1.5, 2.3], [1.7, 2.7], [1.9, 3.1], 
    [2.1, 3.5], [2.3, 3.9], [2.5, 4.3], [2.7, 4.7], [2.9, 5.1]]
Y = [2.5, 3.2, 3.9, 4.6, 5.3, 6, 6.7, 7.4, 8.1, 8.8]

TestGD(GradientDescent, X, Y, 0.1, 50000)
TestGD(GradientDescent2, X, Y, 0.1, 50000)
TestGD(GradientDescent3, X, Y, 0.1, 50000)


