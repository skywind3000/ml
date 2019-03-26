//=====================================================================
//
// ineuron.c - Neural Network C Implementation
//
// Created by skywind on 2008/08/27
// Last Modified: 2019/03/26 22:41:34
//
//=====================================================================
#include "ineuron.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


//=====================================================================
// Neuron Function Callbacks
//=====================================================================
static double ineuron_normal_function(double x)
{
	return x;
}

static double ineuron_normal_derivative(double y)
{
	return 1;
}

static double ineuron_sigmod_function(double x)
{
	return (1 / (1 + exp(-2 * x)));
}

static double ineuron_sigmod_derivative(double y)
{
	return (2 * y * (1 - y));
}

static double ineuron_bipolar_function(double x)
{
	return ((2 / (1 + exp(-2 * x))) - 1);
}

static double ineuron_bipolar_derivative(double y)
{
	return (2 * (1 - y * y) / 2);
}

static double ineuron_threshold_function(double x)
{
	return (x >= 0) ? 1 : 0;
}

static double ineuron_threshold_derivative(double y)
{
	return 0;
}

static double ineuron_tanh_function(double x)
{
	return tanh(x);
}

static double ineuron_tanh_derivative(double y)
{
	return 1.0 - y * y;
}


//=====================================================================
// simple randomize function
//=====================================================================
static int _inseed = 0;

static int _irandint(void) 
{
	_inseed = _inseed * 214013 + 2531011;
	return ((_inseed >> 16) & 0x7fff);
}

static double _irandom(void)
{
	return ((double)_irandint()) / 0x8000;
}


//=====================================================================
// NEURON INTERFACES
//=====================================================================

//---------------------------------------------------------------------
// ineuron_create - create a new neuron
//   mode = INEURON_FUN_NORMAL ->   f(x) = x
//   mode = INEURON_FUN_SIGMOD ->   f(x) = ((2 / (1 + exp(-2 * x))) - 1);
// you can change the callbacks manually for other functions
// you can change the callbacks fmanually or other functions
//---------------------------------------------------------------------
struct INEURON *ineuron_create(int input_count, double threshold, int mode)
{
	struct INEURON *neuron;
	unsigned long size;
	int i;

	assert(input_count > 0);
	size = sizeof(struct INEURON) + sizeof(double) * input_count;
	neuron = (struct INEURON*)malloc(size);

	assert(neuron != NULL);

	neuron->input_count = input_count;
	neuron->neuron_mode = INEURON_MODE_ACTIVATION;
	neuron->threshold = threshold;
	neuron->output = 0;

	neuron->function = NULL;
	neuron->derivative = NULL;

	switch (mode) 
	{
	case INEURON_FUN_NORMAL:
		neuron->function = ineuron_normal_function;
		neuron->derivative = ineuron_normal_derivative;
		break;
	case INEURON_FUN_SIGMOD:
		neuron->function = ineuron_sigmod_function;
		neuron->derivative = ineuron_sigmod_derivative;
		break;
	case INEURON_FUN_BIPOLAR:
		neuron->function = ineuron_bipolar_function;
		neuron->derivative = ineuron_bipolar_derivative;
		break;
	case INEURON_FUN_THRESHOLD:
		neuron->function = ineuron_threshold_function;
		neuron->derivative = ineuron_threshold_derivative;
		break;
	case INEURON_FUN_TANH:
		neuron->function = ineuron_tanh_function;
		neuron->derivative = ineuron_tanh_derivative;
		break;
	}

	for (i = 0; i < input_count; i++) {
		neuron->weight[i] = 0;
	}

	return neuron;
}

//---------------------------------------------------------------------
// ineuron_destroy - destroy a neuron
//---------------------------------------------------------------------
void ineuron_destroy(struct INEURON *neuron)
{
	assert(neuron != NULL);
	neuron->function = NULL;
	free(neuron);
}

//---------------------------------------------------------------------
// ineuron_randomize - randomize weight
//---------------------------------------------------------------------
void ineuron_randomize(struct INEURON *neuron, double range, double min)
{
	int i;

	assert(neuron);
	for (i = neuron->input_count - 1; i >= 0; i--) {
		neuron->weight[i] = _irandom() * range + min;
	}
	neuron->threshold = _irandom() * range + min;
}

//---------------------------------------------------------------------
// ineuron_destroy - compute inputs
//---------------------------------------------------------------------
void ineuron_compute(struct INEURON *neuron, const double *inputs)
{
	double output = 0;
	double *weight, value;
	int i;

	assert(neuron && inputs);
	weight = neuron->weight;

	if (neuron->neuron_mode == INEURON_MODE_ACTIVATION) {
		for (i = neuron->input_count - 1; i >= 0; i--) {
			output += inputs[i] * weight[i];
			//printf("%f ", weight[i]);
		}
		output = neuron->function(output + neuron->threshold);
	}	
	else if (neuron->neuron_mode == INEURON_MODE_DISTANCE) {
		for (i = neuron->input_count - 1; i >= 0; i--) {
			value = inputs[i] - weight[i];
			output += (value >= 0)? value : (-value);
		}
	}
	//printf("->%f\n", output + neuron->threshold);

	neuron->output = output;
}


//=====================================================================
// NEURAL INTERFACES
//=====================================================================

//---------------------------------------------------------------------
// inlayer_create - create a activation nerual layer
//---------------------------------------------------------------------
struct INLAYER *inlayer_create(int neuron_count, int input_count, int mode)
{
	struct INLAYER *layer;
	unsigned long size;
	int i;

	assert(neuron_count > 0 && input_count > 0);

	size = sizeof(struct INLAYER) + sizeof(struct INEURON*) * neuron_count;
	layer = (struct INLAYER*)malloc(size);
	assert(layer);

	layer->output = (double*)malloc(sizeof(double) * neuron_count);
	assert(layer->output);

	layer->input_count = input_count;
	layer->neuron_count = neuron_count;

	for (i = 0; i < neuron_count; i++) {
		layer->neuron[i] = ineuron_create(input_count, 1, mode);
		layer->output[i] = 0;
	}

	return layer;
}


//---------------------------------------------------------------------
// inlayer_destroy - destroy a activation neural layer
//---------------------------------------------------------------------
void inlayer_destroy(struct INLAYER *layer)
{
	int i;

	assert(layer);
	assert(layer->output);

	for (i = 0; i < layer->neuron_count; i++) {
		ineuron_destroy(layer->neuron[i]);
		layer->neuron[i] = NULL;
	}

	free(layer->output);
	layer->output = NULL;

	free(layer);
}


//---------------------------------------------------------------------
// inlayer_randmize - randomize
//---------------------------------------------------------------------
void inlayer_randomize(struct INLAYER *layer, double range, double min)
{
	int i;
	assert(layer);
	for (i = 0; i < layer->neuron_count; i++) {
		ineuron_randomize(layer->neuron[i], range, min);
	}
}

//---------------------------------------------------------------------
// inlayer_compute - compute inputs
//---------------------------------------------------------------------
void inlayer_compute(struct INLAYER *layer, const double *inputs)
{
	int i;

	assert(layer);

	for (i = layer->neuron_count - 1; i >= 0; i--) {
		ineuron_compute(layer->neuron[i], inputs);
		layer->output[i] = layer->neuron[i]->output;
	}
}

//---------------------------------------------------------------------
// inlayer_compute_mode - change neuron compute mode
// mode = INEURON_MODE_ACTIVATION  ->  using activation computing
// mode = INEURON_MODE_DISTANCE    ->  using distance computing
//---------------------------------------------------------------------
void inlayer_compute_mode(struct INLAYER *layer, int mode)
{
	int i;
	for (i = 0; i < layer->neuron_count; i++) {
		layer->neuron[i]->neuron_mode = mode;
	}
}

//---------------------------------------------------------------------
// inlayer_function - set function
//---------------------------------------------------------------------
// inlayer_function - set function
void inlayer_function(struct INLAYER *layer, 
						iNeuronFn function,
						iNeuronFn derivative)
{
	int i;
	assert(layer);
	for (i = 0; i < layer->neuron_count; i++) {
		layer->neuron[i]->function = function;
		layer->neuron[i]->derivative = derivative;
	}
}


//=====================================================================
// INEURALNET - activation neural network
//=====================================================================

//---------------------------------------------------------------------
// inn_create - create a neural network
//---------------------------------------------------------------------
struct INEURALNET *inn_create(int *layervec, int input_count, int mode)
{
	struct INEURALNET *net;
	unsigned long size;
	int layer_count;
	int count, i;

	assert(layervec);
	for (i = 0; layervec[i] > 0; i++);
	layer_count = i;

	assert(layer_count > 0);
	size = sizeof(struct INEURALNET) + sizeof(struct INLAYER*) * layer_count;
	net = (struct INEURALNET*)malloc(size);

	assert(net);
	net->layer_count = layer_count;
	net->input_count = input_count;
	net->output_count = layervec[layer_count - 1];

	net->output = (double*)malloc(sizeof(double) * net->output_count);
	assert(net->output);

	for (i = 0; i < layer_count; i++) {
		count = input_count;
		if (i > 0) count = layervec[i - 1];
		net->layer[i] = inlayer_create(layervec[i], count, mode);
	}

	for (i = 0; i < net->output_count; i++) {
		net->output[i] = 0;
	}

	return net;
}

//---------------------------------------------------------------------
// inn_destroy - destroy a neural network
//---------------------------------------------------------------------
void inn_destroy(struct INEURALNET *net)
{
	int i;

	assert(net);
	assert(net->output);

	for (i = 0; i < net->layer_count; i++) {
		inlayer_destroy(net->layer[i]);
		net->layer[i] = NULL;
	}

	free(net->output);
	net->output = 0;

	free(net);
}


//---------------------------------------------------------------------
// inn_randomize - randomize a network
//---------------------------------------------------------------------
void inn_randomize(struct INEURALNET *net, double range, double min)
{
	int i;

	assert(net);

	for (i = 0; i < net->layer_count; i++) {
		inlayer_randomize(net->layer[i], range, min);
	}
}


//---------------------------------------------------------------------
// inn_compute - compute inputes
//---------------------------------------------------------------------
void inn_compute(struct INEURALNET *net, const double *inputs)
{
	struct INLAYER *layer;
	double *input;
	int i;
	
	for (i = 0; i < net->layer_count; i++) {
		input = (double*)inputs;
		if (i > 0) input = net->layer[i - 1]->output;
		inlayer_compute(net->layer[i], input);
	}

	layer = net->layer[net->layer_count - 1];
	for (i = 0; i < layer->neuron_count; i++) {
		net->output[i] = layer->output[i];
	}
}


//---------------------------------------------------------------------
// inn_compute_mode - change neuron compute mode
// mode = INEURON_MODE_ACTIVATION  ->  using activation computing
// mode = INEURON_MODE_DISTANCE    ->  using distance computing
//---------------------------------------------------------------------
void inn_compute_mode(struct INEURALNET *net, int mode)
{
	int i;
	assert(net);
	for (i = 0; i < net->layer_count; i++) {
		inlayer_compute_mode(net->layer[i], mode);
	}
}

//---------------------------------------------------------------------
// inn_function - set function
//---------------------------------------------------------------------
void inn_function(struct INEURALNET *net, 
					iNeuronFn function,
					iNeuronFn derivative)
{
	int i;
	assert(net);
	for (i = 0; i < net->layer_count; i++) {
		inlayer_function(net->layer[i], function, derivative);
	}
}




//=====================================================================
// INTRAIN_BP - back propagation learning
//=====================================================================

//---------------------------------------------------------------------
// intrain_bp_create - create a back propagation learning class
//---------------------------------------------------------------------
struct INTRAIN_BP *intrain_bp_create(struct INEURALNET *net, double rate)
{
	struct INTRAIN_BP *bp;
	struct INLAYER *layer;
	int nlayer, ncount;
	int size, i, j, k;

	bp = (struct INTRAIN_BP*)malloc(sizeof(struct INTRAIN_BP));
	assert(net && bp);

	bp->network = net;
	bp->layer_count = net->layer_count;
	bp->input_count = net->input_count;
	bp->output_count = net->output_count;
	
	bp->learning_rate = rate;
	bp->momentum = 0;

	nlayer = net->layer_count;

	bp->neuron_error = (double**)malloc(sizeof(double*) * nlayer);
	bp->weight_update = (double***)malloc(sizeof(double**) * nlayer);
	bp->threshold_update = (double**)malloc(sizeof(double*) * nlayer);

	assert(bp->neuron_error && bp->weight_update && bp->threshold_update);

	for (i = 0; i < nlayer; i++) {
		layer = net->layer[i];
		ncount = layer->neuron_count;
		bp->neuron_error[i] = (double*)malloc(sizeof(double) * ncount);
		bp->weight_update[i] = (double**)malloc(sizeof(double*) * ncount);
		bp->threshold_update[i] = (double*)malloc(sizeof(double) * ncount);

		assert(bp->neuron_error[i]);
		assert(bp->weight_update[i]);
		assert(bp->threshold_update[i]);

		for (j = 0; j < ncount; j++) {
			bp->neuron_error[i][j] = 0;
			bp->threshold_update[i][j] = 0;

			size = sizeof(double) * layer->neuron[j]->input_count;
			bp->weight_update[i][j] = (double*)malloc(size);
			assert(bp->weight_update[i][j]);

			for (k = 0; k < layer->neuron[j]->input_count; k++) {
				bp->weight_update[i][j][k] = 0;
			}
		}
	}

	return bp;
}


//---------------------------------------------------------------------
// intrain_bp_destroy - destroy a back propagation learning class
//---------------------------------------------------------------------
void intrain_bp_destroy(struct INTRAIN_BP *bp)
{
	struct INLAYER *layer;
	int i, j;

	assert(bp);
	assert(bp->neuron_error && bp->weight_update && bp->threshold_update);
	
	for (i = 0; i < bp->layer_count; i++) {
		layer = bp->network->layer[i];
		assert(bp->neuron_error[i]);
		assert(bp->weight_update[i]);
		assert(bp->threshold_update[i]);

		for (j = 0; j < layer->neuron_count; j++) {
			free(bp->weight_update[i][j]);
		}

		free(bp->neuron_error[i]);
		free(bp->weight_update[i]);
		free(bp->threshold_update[i]);
	}

	free(bp->neuron_error);
	free(bp->weight_update);
	free(bp->threshold_update);
	
	bp->neuron_error = NULL;
	bp->weight_update = NULL;
	bp->threshold_update = NULL;

	free(bp);
}


//---------------------------------------------------------------------
// calculate bp errors
//---------------------------------------------------------------------
static double intrain_bp_calculate_error(struct INTRAIN_BP *bp, 
										const double *desired)
{
	struct INEURALNET *net;
	struct INLAYER *layer;
	struct INLAYER *layer_next;
	struct INEURON *neuron;
	double *errors;
	double *errors_next;
	double output;
	double error;
	double sum, e;
	int nlayer;
	int i, j, k;

	net = bp->network;
	nlayer = net->layer_count;
	layer = net->layer[nlayer - 1];
	errors = bp->neuron_error[nlayer - 1];
	error = 0;

	for (i = layer->neuron_count - 1; i >= 0; i--) {
		neuron = layer->neuron[i];
		output = neuron->output;
		e = desired[i] - output;
		errors[i] = e * neuron->derivative(output);
		error += e * e;
	}

	for (j = nlayer - 2; j >= 0; j--) {
		layer = net->layer[j];
		layer_next = net->layer[j + 1];
		errors = bp->neuron_error[j];
		errors_next = bp->neuron_error[j + 1];
		for (i = layer->neuron_count - 1; i >= 0; i--) {
			sum = 0;
			neuron = layer->neuron[i];
			for (k = layer_next->neuron_count - 1; k >= 0; k--) {
				sum += errors_next[k] * layer_next->neuron[k]->weight[i];
			}
			errors[i] = sum * neuron->derivative(neuron->output);
		}
	}

	return error / 2;
}


//---------------------------------------------------------------------
// calculate bp update
//---------------------------------------------------------------------
static void intrain_bp_calculate_update(struct INTRAIN_BP *bp, 
									const double *desired)
{
	struct INEURALNET *net;
	struct INLAYER *layer;
	struct INEURON *neuron;
	double **weight_update;
	double *neuron_weight_update;
	double *threshold_update;
	double *errors;
	double *inputs;
	double learning_rate;
	double momentum;
	double error;
	int i, j, k;

	net = bp->network;
	learning_rate = bp->learning_rate;
	momentum = bp->momentum;

	for (k = 0; k < net->layer_count; k++) {
		layer = net->layer[k];

		errors = bp->neuron_error[k];
		weight_update = bp->weight_update[k];
		threshold_update = bp->threshold_update[k];

		inputs = (k == 0)? (double*)desired : net->layer[k - 1]->output;

		for (i = layer->neuron_count - 1; i >= 0; i--) {
			neuron = layer->neuron[i];
			error = errors[i];
			neuron_weight_update = weight_update[i];
			for (j = neuron->input_count - 1; j >= 0; j--) {
				neuron_weight_update[j] = learning_rate * (momentum *
					neuron_weight_update[j] + (1.0 - momentum) * 
					error * inputs[j]);
			}
			threshold_update[i] = learning_rate * (momentum * 
				threshold_update[i] + (1.0 - momentum) * error);
		}
	}
}


//---------------------------------------------------------------------
// update weights
//---------------------------------------------------------------------
static void intrain_bp_update_network(struct INTRAIN_BP *bp)
{
	struct INEURALNET *net;
	struct INLAYER *layer;
	struct INEURON *neuron;
	double **weight_update;
	double *threshold_update;
	double *neuron_weight_update;
	int i, j, k;

	net = bp->network;
	for (i = 0; i < net->layer_count; i++) {
		layer = net->layer[i];
		weight_update = bp->weight_update[i];
		threshold_update = bp->threshold_update[i];

		for (j = layer->neuron_count - 1; j >= 0; j--) {
			neuron = layer->neuron[j];
			neuron_weight_update = weight_update[j];
			for (k = neuron->input_count - 1; k >= 0; k--) {
				neuron->weight[k] += neuron_weight_update[k];
			}
			neuron->threshold += threshold_update[j];
		}
	}
}


//---------------------------------------------------------------------
// intrain_bp_run - run learning
//---------------------------------------------------------------------
double intrain_bp_run(struct INTRAIN_BP *bp, const double *input, 
		const double *desired_output)
{
	double error;

	inn_compute(bp->network, input);

	error = intrain_bp_calculate_error(bp, desired_output);

	intrain_bp_calculate_update(bp, input);
	intrain_bp_update_network(bp);

	return error;
}

//---------------------------------------------------------------------
// intrain_bp - run bp to learn all (input, output)
//---------------------------------------------------------------------
int intrain_bp_epoch(struct INEURALNET *net, const double **inputv, const
		double **outputv, double learnrate, int times, double limit)
{
	struct INTRAIN_BP *bp;
	double *inputs;
	double *outputs;
	double error;
	int i;

	assert(net && inputv && outputv);

	bp = intrain_bp_create(net, learnrate);
	assert(bp);

	inputs = (double*)inputv[0];
	outputs = (double*)outputv[0];

	for (i = 0; i < times && inputs && outputs; i++, inputs++, outputs++) {
		error = intrain_bp_run(bp, inputs, outputs);
		if (error < limit) break;
	}

	intrain_bp_destroy(bp);

	return i;
}


