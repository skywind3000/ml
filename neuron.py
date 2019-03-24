#! /usr/bin/env python
# -*- coding: utf-8 -*
#======================================================================
# 
# neuron.py - Neuron Network Implementation
#
# NOTE:
# for more information, please see the readme file.
#
#======================================================================
from __future__ import print_function
import math
import sys
import random


#----------------------------------------------------------------------
# 2/3 compatible
#----------------------------------------------------------------------
if sys.version_info[0] >= 3:
    xrange = range


#----------------------------------------------------------------------
# Function 
#----------------------------------------------------------------------
FN_NORMAL       = "normal"      # f(x) = x
FN_SIGMOID      = "sigmoid"     # f(x) = 1 / (1 + exp(-2 * x))
FN_BIPOLAR      = "bipolar"     # f(x) = 2 / (1 + exp(-2 * x)) - 1
FN_THRESHOLD    = "threshold"   # f(x) = (x >= 0) ? 1 : 0
FN_TANH         = "tanh"        # f(x) = tanh(x)

fn_normal_function = lambda x: x
fn_normal_derivative = lambda y: 1.0

fn_sigmoid_function = lambda x: 1.0 / (1.0 + math.exp(-2.0 * x))
fn_sigmoid_derivative = lambda y: 2.0 * y * (1.0 - y)

fn_bipolar_function = lambda x: (2.0 / (1.0 + math.exp(-2.0 * x))) - 1.0
fn_bipolar_derivative = lambda y: 1.0 - y * y

fn_threshold_function = lambda x: (x >= 0) and 1.0 or 0.0
fn_threshold_derivative = lambda y: 0

fn_tanh_function = lambda x: math.tanh(x)
fn_tanh_derivative = lambda y: 1.0 - y * y

FUNCTION_MAP = {
    FN_NORMAL: (fn_normal_function, fn_normal_derivative),
    FN_SIGMOID: (fn_sigmoid_function, fn_sigmoid_derivative),
    FN_BIPOLAR: (fn_bipolar_function, fn_bipolar_derivative),
    FN_THRESHOLD: (fn_threshold_function, fn_threshold_derivative),
    FN_TANH: (fn_tanh_function, fn_tanh_derivative)
}

#----------------------------------------------------------------------
# Mode 
#----------------------------------------------------------------------
NEURON_MODE_ACTIVATION  = 0
NEURON_MODE_DISTANCE    = 1


#----------------------------------------------------------------------
# NEURON
#----------------------------------------------------------------------
class Neuron (object):

    def __init__ (self, input_count, fn = FN_NORMAL):
        self.input_count = input_count
        self.weight = [0.0] * input_count
        self.threshold = 0.0
        self.mode = NEURON_MODE_ACTIVATION
        self.setup(fn)
        self.output = 0.0
    
    def setup (self, fn):
        self.function = fn_normal_function
        self.derivative = fn_normal_derivative
        cb = FUNCTION_MAP.get(fn.lower(), None)
        if cb is not None:
            self.function = cb[0]
            self.derivative = cb[1]
        self.name = fn.lower()
    
    def __len__ (self):
        return self.input_count
    
    def __getitem__ (self, k):
        return self.weight[k]
    
    def __setitem__ (self, k, v):
        self.weight[k] = v
    
    def __repr__ (self):
        return 'Neuron(%d)'%self.input_count
    
    def __str__ (self):
        return self.__repr__()
    
    def randomize (self, range = 1.0, minimal = 0.0):
        for i in xrange(self.input_count):
            self.weight[i] = random.random() * range + minimal
        self.threshold = random.random() * range + minimal
    
    def reset (self):
        for i in xrange(self.input_count):
            self.weight[i] = 0
        self.threshold = 0
    
    def compute (self, inputs):
        weight = self.weight
        output = 0.0
        if len(inputs) != self.input_count:
            raise ValueError('input size error')
        if self.mode == NEURON_MODE_ACTIVATION:
            for i in xrange(self.input_count):
                output += weight[i] * inputs[i]
            output = self.function(output + self.threshold)
        else:
            for i in xrange(self.input_count):
                output += abs(inputs[i] - weight[i])
        self.output = output
        return output
    
    def text (self):
        return 'Weight(' + ','.join([ '%.3f'%w for w in self.weight ]) + ')'

    

#----------------------------------------------------------------------
# Layer
#----------------------------------------------------------------------
class Layer (object):

    def __init__ (self, neuron_count, input_count, fn = FN_NORMAL):
        self.neuron = []
        for i in xrange(neuron_count):
            neuron = Neuron(input_count, fn)
            neuron.threshold = 1.0
            self.neuron.append(neuron)
        self.neuron_count = neuron_count
        self.input_count = input_count
        self.output = [0.0] * neuron_count
        self.neuron = tuple(self.neuron)
    
    def __len__ (self):
        return self.neuron_count
    
    def __getitem__ (self, k):
        return self.neuron[k]
    
    def __repr__ (self):
        return 'Layer(%d, %d)'%(self.neuron_count, self.input_count)
    
    def __str__ (self):
        return self.__repr__()
    
    def __iter__ (self):
        return self.neuron.__iter__()

    def mode (self, newmode):
        for neuron in self.neuron:
            neuron.mode = newmode
    
    def function (self, function, derivative):
        for neuron in self.neuron:
            neuron.function = function
            neuron.derivative = derivative

    def randomize (self, range = 1.0, minimal = 0.0):
        for neuron in self.neuron:
            neuron.randomize(range, minimal)
    
    def reset (self):
        for neuron in self.neuron:
            neuron.reset()
    
    def compute (self, inputs):
        if len(inputs) != self.input_count:
            raise ValueError('input size error')
        for i in xrange(self.neuron_count):
            neuron = self.neuron[i]
            neuron.compute(inputs)
            self.output[i] = neuron.output
        return self.output
    
    def text (self):
        text = ',\n'.join([ '    ' + n.text() for n in self.neuron ]) + '.\n'
        return self.__repr__() + ':\n' + text
    

#----------------------------------------------------------------------
# Network
#----------------------------------------------------------------------
class Network (object):

    def __init__ (self, layer_vector, input_count, fn = FN_NORMAL):
        self.layer = []
        self.layer_count = len(layer_vector)
        if len(layer_vector) < 1:
            raise ValueError('layer vector error')
        for i in xrange(self.layer_count):
            count = input_count
            if i > 0: 
                count = layer_vector[i - 1]
            layer = Layer(layer_vector[i], count, fn)
            self.layer.append(layer)
        self.input_count = input_count
        self.output_count = layer_vector[-1]
        self.layer_vector = tuple(layer_vector)
        self.output = [0.0] * self.output_count
        self.layer = tuple(self.layer)
    
    def __len__ (self):
        return self.layer_count
    
    def __getitem__ (self, k):
        return self.layer[k]
    
    def __repr__ (self):
        return 'Network(%s, %d)'%(self.layer_vector, self.input_count)
    
    def __str__ (self):
        return self.__repr__()
    
    def __iter__ (self):
        return self.layer.__iter__()
    
    def randomize (self, range = 2.0, minimal = 1):
        for layer in self.layer:
            layer.randomize(range, minimal)
    
    def reset (self):
        for layer in self.layer:
            layer.reset()
    
    def mode (self, newmode):
        for layer in self.layer:
            layer.mode(newmode)

    def function (self, function, derivative):
        for layer in self.layer:
            layer.function(function, derivative)
    
    def compute (self, inputs):
        if len(inputs) != self.input_count:
            raise ValueError('input size error')
        for i in xrange(self.layer_count):
            vector = inputs
            if i > 0:
                vector = self.layer[i - 1].output
            self.layer[i].compute(vector)
        layer = self.layer[-1]
        for i in xrange(self.output_count):
            self.output[i] = layer.output[i]
        return self.output
    
    def text (self):
        text = '\n'.join([ l.text() for l in self.layer ])
        return text



#----------------------------------------------------------------------
# Back Propagation
#----------------------------------------------------------------------
class BP (object):

    def __init__ (self, network, learning_rate = 0.5, momentum = 0.1):
        self.network = network
        self.layer_count = network.layer_count
        self.input_count = network.input_count
        self.output_count = network.output_count
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.neuron_error = [ [] for i in xrange(self.layer_count) ]
        self.weight_update = [ [] for i in xrange(self.layer_count) ]
        self.threshold_update = [ [] for i in xrange(self.layer_count) ]
        for i in xrange(self.layer_count):
            layer = self.network[i]
            count = len(layer)
            self.neuron_error[i] = [0.0] * count
            self.threshold_update[i] = [0.0] * count
            self.weight_update[i] = [ [] for c in xrange(count) ]
            for j in xrange(count):
                self.weight_update[i][j] = [0.0] * layer.input_count
            self.weight_update[i] = tuple(self.weight_update[i])
        self.neuron_error = tuple(self.neuron_error)
        self.threshold_update = tuple(self.threshold_update)

    def __update_error (self, desired):
        net = self.network
        nlayer = self.layer_count
        layer = net.layer[-1]
        errors = self.neuron_error[-1]
        error = 0
        for i in xrange(layer.neuron_count):
            neuron = layer.neuron[i]
            output = neuron.output
            e = desired[i] - output
            errors[i] = e * neuron.derivative(output)
            error += e * e
        for p in xrange(nlayer - 1):
            j = nlayer - 2 - p
            layer = net.layer[j]
            layer_next = net.layer[j + 1]
            errors = self.neuron_error[j]
            errors_next = self.neuron_error[j + 1]
            for i in xrange(layer.neuron_count):
                neuron = layer.neuron[i]
                sum = 0
                t1, t2 = 0, 0
                for k in xrange(layer_next.neuron_count):
                    t1 += errors_next[k]
                    t2 += layer_next.neuron[k].weight[i]
                    sum += errors_next[k] * layer_next.neuron[k].weight[i]
                #print 'sum', sum, t1, t2
                errors[i] = sum * neuron.derivative(neuron.output)
        #print 'error', self.neuron_error
        return error * 0.5
    
    def __update_calculation (self, desired):
        net = self.network
        learning_rate = self.learning_rate
        momentum = self.momentum
        for k in xrange(net.layer_count):
            layer = net.layer[k]
            errors = self.neuron_error[k]
            weight_update = self.weight_update[k]
            threshold_update = self.threshold_update[k]
            inputs = desired 
            if k > 0: 
                inputs = net.layer[k - 1].output
            for i in xrange(layer.neuron_count):
                neuron = layer.neuron[i]
                error = errors[i]
                update = weight_update[i]
                for j in xrange(neuron.input_count):
                    u = momentum * update[j]
                    v = (1 - momentum) * error * inputs[j]
                    update[j] = learning_rate * (u + v)
                u = momentum * threshold_update[i]
                v = (1.0 - momentum) * error
                threshold_update[i] = learning_rate * (u + v)
        return 0
    
    def __update_network (self):
        net = self.network
        for i in xrange(self.layer_count):
            layer = net.layer[i]
            weight_update = self.weight_update[i]
            threshold_update = self.threshold_update[i]
            for j in xrange(layer.neuron_count):
                neuron = layer.neuron[j]
                update = weight_update[j]
                for k in xrange(neuron.input_count):
                    neuron.weight[k] += update[k]
                neuron.threshold += threshold_update[j]
        return 0
    
    def run (self, input, desired_output):
        self.network.compute(input)
        error = self.__update_error(desired_output)
        self.__update_calculation(input)
        self.__update_network()
        return error


#----------------------------------------------------------------------
# NN
#----------------------------------------------------------------------
class NN (object):

    def __init__ (self, ni, nh, no, fn = FN_NORMAL):
        self.network = Network([nh, no], ni, fn)
        self.input_count = ni
        self.output_count = no
        self.layer_count = 2
        self.network.function(fn_tanh_function, fn_tanh_derivative)
        self.network.randomize()
    
    def compute (self, inputs):
        return self.network.compute(inputs)
    
    def train (self, patterns, times = 10000, limit = 0.0, N = 0.5, M = 0.1):
        bp = BP(self.network, N, M)
        error = 0.0
        for i in xrange(times):
            error = 0.0
            for inputs, desired_outputs in patterns:
                error += bp.run(inputs, desired_outputs)
            if error < limit:
                break
            if i % 100 == 0:
                print('error', error, i)
        return error
    
    def test (self, patterns):
        for p in patterns:
            print(p[0], '->', self.compute(p[0]))


#----------------------------------------------------------------------
# testing case
#----------------------------------------------------------------------
if __name__ == '__main__':
    def test0():
        y = Layer(2, 2)
        y.randomize()
        print(y.text())
    def test1():
        n = Network([5, 10], 2)
        n.randomize()
        print(n.compute([1,2]))
        print(n.text())
        return 0
    def test2():
        # Teach network XOR function
        pat = [
            [[0,0], [0]],
            [[0,1], [1]],
            [[1,0], [1]],
            [[1,1], [0]]
        ]
        n = NN(2, 2, 1)
        #n.network.randomize()
        n.train(pat, 90000)
        n.test(pat)
        print('')
        print(n.network.text())

    test2()

