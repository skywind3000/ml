//=====================================================================
//
// ksvm.h - SVM Implementation
//
// Created by skywind on 2019/03/28
// Last Modified: 2019/03/28 15:35:47
//
//=====================================================================
#ifndef _KSVM_H_
#define _KSVM_H_

#include <stddef.h>


// SVM type
#define KSVM_TYPE_SVC        0
#define KSVM_TYPE_SVR        1

// kernel types
#define KSVM_KERNEL_LINEAR        0
#define KSVM_KERNEL_POLY          1
#define KSVM_KERNEL_RBF           2
#define KSVM_KERNEL_SIGMOID       3
#define KSVM_KERNEL_PRECOMPUTED   4
#define KSVM_KERNEL_USER          5


// readonly vector
typedef struct {
	int uid;          // row id in problem (used for precompute)
	int size;         // feature cound for dense, elements for sparse
	int sparse;       // 0 for dense, 1 for sparse
	int *index;       // index for sparse elements
	double *value;    // values
}	ksvm_vector_t;

// problem: dataset
typedef struct {
	int nrows;            // number of samples
	int capacity;         // capacity
	double *y;            // y values: -1 or 1
	ksvm_vector_t **x;    // x vectors
}	ksvm_problem_t;

// parameters
typedef struct {
	int svm_type;
	int kernel_type;
	int degree;           // for poly
	double gamma;         // for poly/rbf/sigmoid
	double coef0;         // for poly/sigmoid
	double C;             // constraint
	double cache_size;    // in MB
	double eps;           // tolerance
	int p;                // for regression
	int nweights;
	double *weight;
	int *weight_label;
}	ksvm_parameter_t;

// kernel cache
typedef struct {
	int i, j;    // xi
	double value;
}	ksvm_kcache_t;

// kernel
typedef struct {
	int kernel_type;
	int nrows;
	int degree;                  // for poly
	double gamma;                // for poly/rbf/sigmoid
	double coef0;                // for poly/sigmoid
	const ksvm_problem_t *pb;    // problem
	const ksvm_vector_t **x;     // x table
	double *square;              // x square
	void *closure;               // kernel closure
	double *diag;                // diagonal
	ksvm_kcache_t *cache;        // kernel cache
	long csize;                  // cache size
	long cmask;                  // cache mask
	long cache_on;               // statistic on
	long cache_miss;             // statistic missed
	double (*func)(void *, const ksvm_vector_t*, const ksvm_vector_t*);
}	ksvm_kernel_t;

// solver
typedef struct {
	int svm_type;             // KSVM_TYPE_SVC / KSVM_TYPE_SVR
	int *index;               // vector index
	double *y;                // labels
	double *alpha;            // alphas
	int active_size;          // working set size
	int nrows;                // number of samples
	int label_n;              // label negative
	int label_p;              // label positive
	int svn;                  // support vector size
	ksvm_vector_t **svx;      // support vectors
	double *sign;             // sign for regression
	double *svy;              // support vector labels
	double *sva;              // support vector alphas
	double *G;                // G(x) = sum([ai * yi * K(xi, x) for i in N])
	double Cn;                // C negative
	double Cp;                // C positive
	double b;                 // b
	double eps;               // eps
	ksvm_kernel_t *kernel;    // kernel
}	ksvm_solver_t;

// model
typedef struct {
	ksvm_parameter_t param;    // parameter
	int nclasses;              // class names
	int size;                  // total support vector number
	int *labels;               // labels for each class
	int *svp;                  // support vector start position
	int *svn;                  // support vector sizes
	double *rho;               // "b" array
	double *svc;               // support vector coefficients
	ksvm_vector_t **svx;       // support vectors
}	ksvm_model_t;


#ifdef __cplusplus
extern "C" {
#endif

//---------------------------------------------------------------------
// vector interfaces
//---------------------------------------------------------------------

// create dense vector
ksvm_vector_t *ksvm_vector_new_dns(const double *value, int size);

// create sparse vector
ksvm_vector_t *ksvm_vector_new_crs(const int *index, const double *value, int size);

// free vector
void ksvm_vector_free(ksvm_vector_t *vec);

// clone vector
ksvm_vector_t* ksvm_vector_clone(const ksvm_vector_t *vec);

// load vector from string
// dense: space separated numbers, eg. "1.0 2.0 3.0"
// sparse: space separated "index:value" pairs, eg. "0:1.0 2:3.0 10:1000"
ksvm_vector_t* ksvm_vector_load(const char *text);

// dot product
double ksvm_vector_dot(const ksvm_vector_t *x, const ksvm_vector_t *y);


//---------------------------------------------------------------------
// problem interfaces
//---------------------------------------------------------------------

// create new problem in given row
ksvm_problem_t *ksvm_problem_new(void);

// push sample into problem
void ksvm_problem_push(ksvm_problem_t *pb, ksvm_vector_t *x, double y);

// free problem
void ksvm_problem_free(ksvm_problem_t *pb);

// problem group, max group is 16
int ksvm_problem_group(const ksvm_problem_t *pb, int *labels, int *counts);



//---------------------------------------------------------------------
// kernel
//---------------------------------------------------------------------

// kernel constructor
ksvm_kernel_t *ksvm_kernel_new(const ksvm_parameter_t *parameter, const ksvm_problem_t *pb);

// dispose kernel
void ksvm_kernel_free(ksvm_kernel_t *kernel);

// install user function, whose signature must be:
// double (*func)(void *closure, const ksvm_vector_t *x, const ksvm_vector_t *y);
void ksvm_kernel_install(ksvm_kernel_t *kernel, void *func, void *closure);

// call kernel function (for prediction)
double ksvm_kernel_call(const ksvm_kernel_t *kernel, const ksvm_vector_t *x, const ksvm_vector_t *y);

// call kernel function in inner x rows (passed from ksvm_problem_t)
double ksvm_kernel_inner(const ksvm_kernel_t *kernel, int xi, int xj);

// query result from cache (if on) or recaculate it
double ksvm_kernel_query(ksvm_kernel_t *kernel, int xi, int xj);


//---------------------------------------------------------------------
// solver
//---------------------------------------------------------------------

// solver create
ksvm_solver_t* ksvm_solver_new(const ksvm_parameter_t *param, ksvm_kernel_t *kernel);

// free solver
void ksvm_solver_free(ksvm_solver_t *solver);

// resize internal buffer
int ksvm_solver_resize(ksvm_solver_t *solver, int newsize);

// init SVC from two labels
int ksvm_solver_init_svc(ksvm_solver_t *solver, const ksvm_parameter_t *param, int lable_p, int label_n);

// get kernel result
double ksvm_solver_kernel(ksvm_solver_t *solver, int xi, int xj);

// calculate G(xi) = sum([ aj * yj * K(xj, xi) for j in range(active_size) ])
double ksvm_solver_calculate_gx(ksvm_solver_t *solver, int i, int active_size);


//---------------------------------------------------------------------
// model
//---------------------------------------------------------------------

// create new model
ksvm_model_t* ksvm_model_new(const ksvm_parameter_t *param);

// free model
void ksvm_model_free(ksvm_model_t *model);

// resize internal buffer
int ksvm_model_resize(ksvm_model_t *model, int newsize);

// train data
int ksvm_model_train(ksvm_model_t *model, const ksvm_problem_t *pb);

// predict an "x" and output label
double ksvm_model_predict(const ksvm_model_t *model, const ksvm_vector_t *x, double *dec_values);


#ifdef __cplusplus
}
#endif


#endif



