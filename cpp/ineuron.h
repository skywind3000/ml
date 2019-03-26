//=====================================================================
//
// ineuron.h - Neural Network C Implementation
//
// Created by skywind on 2008/08/27
// Last Modified: 2019/03/26 22:41:34
//
//=====================================================================
#ifndef _INEURON_H_
#define _INEURON_H_

#ifdef CONFIG_H
#include "config.h"
#endif


typedef double (*iNeuronFn)(double x);


// output = f( sum(Wn * Xn) )    - default mode
#define INEURON_MODE_ACTIVATION		0

// output = f( sum( ABS(Wn - Xn) ) )
#define INEURON_MODE_DISTANCE		1		


//=====================================================================
// INEURON - activation neuron
//=====================================================================
struct INEURON
{
	int input_count;         // count of input
	int neuron_mode;         // INEURON_MODE_ACTIVATION or DISTANCE
	double threshold;        // threshold input
	double output;           // computing result
	iNeuronFn function;      // f (x) = .... computing function
	iNeuronFn derivative;    // f'(x) = .... derivative function
	double weight[1];        // weights vector
};


// f(x) = x
#define INEURON_FUN_NORMAL      0

// f(x) = (1 / (1 + exp(-2 * x)))
#define INEURON_FUN_SIGMOD      1

// f(x) = ((2 / (1 + exp(-2 * x))) - 1)
#define INEURON_FUN_BIPOLAR     2

// f(x) = (x >= 0) ? 1 : 0
#define INEURON_FUN_THRESHOLD   3

// f(x) = tanh(x)
#define INEURON_FUN_TANH        4


#ifdef __cplusplus
extern "C" {
#endif

// ineuron_create - create a new neuron
//   mode = INEURON_FUN_NORMAL ->   f(x) = x
//   mode = INEURON_FUN_BIPOLAR ->   f(x) = ((2 / (1 + exp(-2 * x))) - 1);
// you should manual change the callbacks for other functions
struct INEURON *ineuron_create(int input_count, double threshold, int mode);

// ineuron_destroy - destroy a neuron
void ineuron_destroy(struct INEURON *neuron);

// ineuron_randomize - randomize weight
void ineuron_randomize(struct INEURON *neuron, double range, double min);

// ineuron_destroy - compute inputs
void ineuron_compute(struct INEURON *neuron, const double *inputs);


#ifdef __cplusplus
}
#endif


//=====================================================================
// INLAYER - activation neural layer
//=====================================================================
struct INLAYER
{
	int neuron_count;
	int input_count;
	double *output;
	struct INEURON *neuron[1];
};


#ifdef __cplusplus
extern "C" {
#endif

// inlayer_create - create a activation neural layer
struct INLAYER *inlayer_create(int neuron_count, int input_count, int mode);

// inlayer_destroy - destroy a activation neural layer
void inlayer_destroy(struct INLAYER *layer);

// inlayer_randmize - randomize
void inlayer_randomize(struct INLAYER *layer, double range, double min);

// inlayer_compute - compute inputs
void inlayer_compute(struct INLAYER *layer, const double *inputs);


// inlayer_compute_mode - change neuron compute mode
// mode = INEURON_MODE_ACTIVATION  ->  using activation computing
// mode = INEURON_MODE_DISTANCE    ->  using distance computing
void inlayer_compute_mode(struct INLAYER *layer, int mode);

// inlayer_function - set neuron compute function
void inlayer_function(struct INLAYER *layer, 
                        iNeuronFn function,
                        iNeuronFn derivative);


#ifdef __cplusplus
}
#endif


//=====================================================================
// INEURALNET - activation neural network
//=====================================================================
struct INEURALNET
{
	int layer_count;
	int input_count;
	int output_count;
	double *output;
	struct INLAYER *layer[1];
};


#ifdef __cplusplus
extern "C" {
#endif

// inn_create - create a neural network
struct INEURALNET *inn_create(int *layervec, int input_count, int mode);

// inn_destroy - destroy a neural network
void inn_destroy(struct INEURALNET *net);

// inn_randomize - randomize a network
void inn_randomize(struct INEURALNET *net, double range, double min);

// inn_compute - compute inputes
void inn_compute(struct INEURALNET *net, const double *inputs);


// inn_compute_mode - change neuron compute mode
// mode = INEURON_MODE_ACTIVATION  ->  using activation computing
// mode = INEURON_MODE_DISTANCE    ->  using distance computing
void inn_compute_mode(struct INEURALNET *net, int mode);


// inn_function - set function
void inn_function(struct INEURALNET *net, 
                    iNeuronFn function,
                    iNeuronFn derivative);


#ifdef __cplusplus
}
#endif



//=====================================================================
// INTRAIN_BP - back propagation learning
//=====================================================================
struct INTRAIN_BP
{
	struct INEURALNET *network;
	int layer_count;
	int input_count;
	int output_count;

	double learning_rate;
	double momentum;

	double **neuron_error;
	double ***weight_update;
	double **threshold_update;
};


#ifdef __cplusplus
extern "C" {
#endif


// intrain_bp_create - create a back propagation learning class
struct INTRAIN_BP *intrain_bp_create(struct INEURALNET *net, double rate);

// intrain_bp_destroy - destroy a back propagation learning class
void intrain_bp_destroy(struct INTRAIN_BP *bp);

// run learning
double intrain_bp_run(struct INTRAIN_BP *bp, const double *input, 
        const double *desired_output);

// run bp to learn all (input, output)
int intrain_bp_epoch(struct INEURALNET *net, const double **inputv, const
        double **outputv, double learnrate, int times, double limit);


#ifdef __cplusplus
}
#endif


#endif


