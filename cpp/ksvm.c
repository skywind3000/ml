//=====================================================================
//
// ksvm.c - SVM Implementation
//
// Created by skywind on 2019/03/28
// Last Modified: 2019/03/28 15:36:23
//
//=====================================================================
#include "ksvm.h"

#define KSVM_VERBOSE		1

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>


//---------------------------------------------------------------------
// INLINE
//---------------------------------------------------------------------
#ifndef INLINE
#if defined(__GNUC__)

#if (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1))
#define INLINE         __inline__ __attribute__((always_inline))
#else
#define INLINE         __inline__
#endif

#elif (defined(_MSC_VER) || defined(__WATCOMC__))
#define INLINE __inline
#else
#define INLINE 
#endif
#endif

#if (!defined(__cplusplus)) && (!defined(inline))
#define inline INLINE
#endif


//---------------------------------------------------------------------
// Memory Management
//---------------------------------------------------------------------
#ifndef KSVM_MALLOC
#define KSVM_MALLOC(size)  malloc(size)
#endif

#ifndef KSVM_FREE
#define KSVM_FREE(ptr)     free(ptr)
#endif

#define ksvm_malloc(size)      KSVM_MALLOC(size)
#define ksvm_calloc(si, count) ksvm_malloc((si) * (count))
#define ksvm_new(type, count)  ((type*)ksvm_calloc(sizeof(type), count))
#define ksvm_free(ptr)         KSVM_FREE(ptr)


//---------------------------------------------------------------------
// library
//---------------------------------------------------------------------
#define ksvm_min(x, y)     (((x) < (y))? (x) : (y))
#define ksvm_max(x, y)     (((x) > (y))? (x) : (y))
#define ksvm_abs(x)        (((x) < 0)? (-(x)) : (x))

static inline double ksvm_powi(double x, int n)
{
	double w = x;
	double z = 1.0;
	for (; n > 0; n >>= 1) {
		if (n & 1) z *= w;
		w = w * w;
	}
	return z;
}

char* ksvm_strip(char *str)
{
	char *ptr = str;
	while (isspace(ptr[0]) && ptr[0]) ptr++;
	if (ptr > str) {
		char *pp = str;
		for (; ptr[0]; ) *pp++ = *ptr++;
		*pp++ = 0;
	}
	ptr = str;
	for (ptr = str; ptr[0]; ptr++);
	while (ptr > str) {
		if (!isspace(ptr[-1])) break;
		ptr[-1] = 0;
		ptr--;
	}
	return str;
}

void *ksvm_realloc(void *ptr, size_t tsize, int oldnum, int newnum)
{
	if (newnum <= 0) {
		if (ptr) ksvm_free(ptr);
		return NULL;	
	}
	else if (ptr == NULL) {
		ptr = ksvm_malloc(tsize * newnum);
		assert(ptr);
		return ptr;
	}
	else {
		void *np = ksvm_malloc(tsize * newnum);
		int minsize = ksvm_min(oldnum, newnum);
		assert(np);
		if (minsize > 0) {
			memcpy(np, ptr, minsize * tsize);
		}
		ksvm_free(ptr);
		return np;
	}
}

static inline double ksvm_f_min(double x, double y)
{
	return (x < y)? x : y;
}

static inline double ksvm_f_max(double x, double y)
{
	return (x > y)? x : y;
}

static inline double ksvm_f_mid(double xmin, double x, double xmax)
{
	return ksvm_f_min(ksvm_f_max(xmin, x), xmax);
}

static inline double ksvm_f_abs(double x)
{
	return (x < 0)? (-x) : x;
}

static inline void ksvm_f_swap(double *x, double *y)
{
	double t = x[0];
	x[0] = y[0];
	y[0] = t;
}

static inline void ksvm_i_swap(int *x, int *y)
{
	int t = x[0];
	x[0] = y[0];
	y[0] = t;
}


//---------------------------------------------------------------------
// create a dense vector
//---------------------------------------------------------------------
ksvm_vector_t *ksvm_vector_new_dns(const double *value, int size)
{
	size_t require = sizeof(ksvm_vector_t) + sizeof(double) * size;
	char *ptr = ksvm_new(char, require);
	ksvm_vector_t *vec = (ksvm_vector_t*)ptr;
	int i;
	if (vec == NULL) return NULL;
	vec->uid = -1;
	vec->size = size;
	vec->sparse = 0;
	vec->index = NULL;
	ptr += sizeof(ksvm_vector_t);
	vec->value = (double*)ptr;
	if (value) {
		memcpy(vec->value, value, size * sizeof(double));
	}	else {
		for (i = 0; i < size; i++) vec->value[i] = 0.0;
	}
	return vec;
}


//---------------------------------------------------------------------
// create a sparse vector
//---------------------------------------------------------------------
ksvm_vector_t *ksvm_vector_new_crs(const int *index, const double *value, int size)
{
	size_t require = sizeof(ksvm_vector_t) + sizeof(double) * size + sizeof(int) * size;
	char *ptr = ksvm_new(char, require);
	ksvm_vector_t *vec = (ksvm_vector_t*)ptr;
	int i;
	if (vec == NULL) return NULL;
	vec->uid = -1;
	vec->size = size;
	vec->sparse = 1;
	ptr += sizeof(ksvm_vector_t);
	vec->value = (double*)ptr;
	ptr += sizeof(double) * size;
	vec->index = (int*)ptr;
	if (value) {
		memcpy(vec->value, value, size * sizeof(double));
	}	else {
		for (i = 0; i < size; i++) vec->value[i] = 0.0;
	}
	if (index) {
		memcpy(vec->index, index, size * sizeof(int));
	}	else {
		for (i = 0; i < size; i++) vec->index[i] = i;
	}
	return vec;
}


//---------------------------------------------------------------------
// free vector
//---------------------------------------------------------------------
void ksvm_vector_free(ksvm_vector_t *vec)
{
	assert(vec->size >= 0);
	vec->uid = -1;
	vec->size = -1;
	vec->sparse = -1;
	vec->index = NULL;
	vec->value = NULL;
	ksvm_free(vec);
}


//---------------------------------------------------------------------
// vector clone
//---------------------------------------------------------------------
ksvm_vector_t* ksvm_vector_clone(const ksvm_vector_t *vec)
{
	ksvm_vector_t *nv; 
	if (vec->sparse == 0) {
		nv = ksvm_vector_new_dns(vec->value, vec->size);
	}	else {
		nv = ksvm_vector_new_crs(vec->index, vec->value, vec->size);
	}
	nv->uid = vec->uid;
	return nv;
}


//---------------------------------------------------------------------
// dot product
//---------------------------------------------------------------------
double (*ksvm_vector_dot_dns)(const ksvm_vector_t *x, const ksvm_vector_t *y) = NULL;
double (*ksvm_vector_dot_crs)(const ksvm_vector_t *x, const ksvm_vector_t *y) = NULL;
double (*ksvm_vector_dot_mix)(const ksvm_vector_t *x, const ksvm_vector_t *y) = NULL;

// dot product
double ksvm_vector_dot(const ksvm_vector_t *x, const ksvm_vector_t *y)
{
	if (x->sparse == 0 && y->sparse == 0) {
		if (ksvm_vector_dot_dns) {
			return ksvm_vector_dot_dns(x, y);
		}
		else {
			int size = ksvm_min(x->size, y->size);
			const double *src = x->value;
			const double *dst = y->value;
			double sum = 0.0;
			for (; size >= 4; src += 4, dst += 4, size -= 4) {
				sum += src[0] * dst[0] + src[1] * dst[1] + 
					src[2] * dst[2] + src[3] * dst[3];
			}
			for (; size > 0; src++, dst++, size--) {
				sum += src[0] * dst[0];
			}
			return sum;
		}
	}	
	else if (x->sparse && y->sparse) {
		if (ksvm_vector_dot_crs) {
			return ksvm_vector_dot_crs(x, y);
		}
		else {
			double sum = 0.0;
			if (x == y) {
				const double *src = x->value;
				int size = x->size;
				for (; size >= 4; src += 4, size -= 4) {
					sum += src[0] * src[0] + src[1] * src[1] + 
						src[2] * src[2] + src[3] * src[3];
				}
				for (; size > 0; src++, size--) {
					sum += src[0] * src[0];
				}
			}
			else {
				const int *ix = x->index;
				const int *iy = y->index;
				const double *px = x->value;
				const double *py = y->value;
				const double *ex = px + x->size;
				const double *ey = py + y->size;
				while (px < ex && py < ey) {
					if (ix[0] == iy[0]) {
						sum += px[0] * py[0];
						px++;
						py++;
						ix++;
						iy++;
					}
					else if (ix[0] > iy[0]) {
						py++;
						iy++;
					}
					else {
						px++;
						ix++;
					}
				}
			}
			return sum;
		}
	}
	else {
		if (x->sparse) {
			const ksvm_vector_t *t = x;
			x = y;
			y = t;
		}
		if (ksvm_vector_dot_mix) {
			return ksvm_vector_dot_mix(x, y);
		}
		else {
			int ix = 0;
			const int *iy = y->index;
			const double *px = x->value;
			const double *py = y->value;
			const double *ex = px + x->size;
			const double *ey = py + y->size;
			double sum = 0.0;
			while (px < ex && py < ey) {
				if (ix == iy[0]) {
					sum += px[0] * py[0];
					px++;
					py++;
					ix++;
					iy++;
				}
				else if (ix > iy[0]) {
					py++;
					iy++;
				}
				else {
					int d = iy[0] - ix;
					px += d;
					ix += d;
				}
			}
			return sum;
		}
	}
}


//---------------------------------------------------------------------
// vector load from string
//---------------------------------------------------------------------
ksvm_vector_t* ksvm_vector_load(const char *text)
{
	const char *ptr = text;
	const char *end = text + strlen(text);
	int sparse = 0;
	int elements = 0;
	while (ptr < end) {
		while (isspace(*ptr) && ptr < end) ptr++;
		if (ptr >= end) break;
		while (isspace(*ptr) == 0 && ptr < end) {
			if (ptr[0] == ':') sparse = 1;
			ptr++;
		}
		elements++;
	}
	if (elements == 0) {
		if (sparse == 0) {
			return ksvm_vector_new_dns(NULL, 0);
		}	else {
			return ksvm_vector_new_crs(NULL, NULL, 0);
		}
	}
	if (sparse == 0) {
		char buffer[80];
		int index = 0;
		ksvm_vector_t *vec = ksvm_vector_new_dns(NULL, elements);
		for (ptr = text; ptr < end; ) {
			int count = 0;
			while (isspace(*ptr) && ptr < end) ptr++;
			if (ptr >= end) break;	
			while (isspace(*ptr) == 0 && ptr < end) {
				if (count < 64) buffer[count++] = ptr[0];
				ptr++;
			}
			buffer[count] = 0;
			if (index < elements) {
				vec->value[index] = atof(buffer);
				index++;
			}
		}
		return vec;
	}
	else {
		char buffer[80];
		int index = 0;
		int last = -1;
		ksvm_vector_t *vec = ksvm_vector_new_crs(NULL, NULL, elements);
		for (ptr = text; ptr < end; ) {
			int count = 0;
			int pos = -1;
			while (isspace(*ptr) && ptr < end) ptr++;
			if (ptr >= end) break;	
			while (isspace(*ptr) == 0 && ptr < end) {
				if (count < 64) {
					buffer[count] = ptr[0];
					if (buffer[count] == ':') pos = count;
					count++;
				}
				ptr++;
			}
			buffer[count] = 0;
			if (index < elements) {
				if (pos < 0) {
					vec->value[index] = atof(buffer);
					vec->index[index] = ++last;
				}	else {
					buffer[pos] = 0;
					vec->index[index] = atoi(buffer);
					vec->value[index] = atof(buffer + pos + 1);
					last = vec->index[index];
				}
				index++;
			}
		}
		return vec;
	}
}


//---------------------------------------------------------------------
// print
//---------------------------------------------------------------------
void ksvm_vector_print(const ksvm_vector_t *x)
{
#if KSVM_VERBOSE
	if (x->sparse == 0) {
		printf("(");
		int i;
		for (i = 0; i < x->size; i++) {
			if (floor(x->value[i]) == x->value[i] && 
				x->value[i] >= -0x7fffffff && 
				x->value[i] <= 0x7fffffff) {
				printf("%ld", (long)x->value[i]);
			}	else {
				printf("%f", x->value[i]);
			}
			if (i + 1 < x->size) printf(", ");
		}
		printf(")\n");
	}
	else {
		printf("{");
		int i;
		for (i = 0; i < x->size; i++) {
			if (floor(x->value[i]) == x->value[i] && 
				x->value[i] >= -0x7fffffff && 
				x->value[i] <= 0x7fffffff) {
				printf("%d:%ld", x->index[i], (long)x->value[i]);
			}	else {
				printf("%d:%f", x->index[i], x->value[i]);
			}
			if (i + 1 < x->size) printf(", ");
		}
		printf("}\n");
	}
#endif
}


//---------------------------------------------------------------------
// create problem
//---------------------------------------------------------------------
ksvm_problem_t *ksvm_problem_new(void)
{
	ksvm_problem_t *pb = ksvm_new(ksvm_problem_t, 1);
	pb->nrows = 0;
	pb->capacity = 0;
	pb->x = NULL;
	pb->y = NULL;
	return pb;
}


//---------------------------------------------------------------------
// push sample
//---------------------------------------------------------------------
void ksvm_problem_push(ksvm_problem_t *pb, ksvm_vector_t *x, double y)
{
	int index = pb->nrows;
	if (pb->nrows >= pb->capacity) {
		int rows = pb->nrows + 1;
		ksvm_vector_t **x;
		double *y;
		pb->capacity = ksvm_max(pb->capacity, 8);
		for (; pb->capacity < rows; pb->capacity *= 2);
		x = ksvm_new(ksvm_vector_t*, pb->capacity);
		y = ksvm_new(double, pb->capacity);
		assert(x);
		assert(y);
		if (pb->nrows > 0) {
			memcpy(x, pb->x, sizeof(ksvm_vector_t*) * pb->nrows);	
			memcpy(y, pb->y, sizeof(double) * pb->nrows);
		}
		ksvm_free(pb->x);
		ksvm_free(pb->y);
		pb->x = x;
		pb->y = y;
	}
	pb->nrows++;
	pb->x[index] = x;
	pb->y[index] = y;
	x->uid = index;
}


//---------------------------------------------------------------------
// problem free
//---------------------------------------------------------------------
void ksvm_problem_free(ksvm_problem_t *pb)
{
	assert(pb);
	if (pb->nrows) {
		int i;
		for (i = 0; i < pb->nrows; i++) {
			ksvm_vector_free(pb->x[i]);
			pb->x[i] = NULL;
		}
	}
	if (pb->x) {
		ksvm_free(pb->x);
		pb->x = NULL;
	}
	if (pb->y) {
		ksvm_free(pb->y);
		pb->y = NULL;
	}
	pb->capacity = 0;
	pb->nrows = 0;
	ksvm_free(pb);
}


//---------------------------------------------------------------------
// problem print
//---------------------------------------------------------------------
void ksvm_problem_print(const ksvm_problem_t *ds)
{
	int i;
	for (i = 0; i < ds->nrows; i++) {
		if (floor(ds->y[i]) == ds->y[i]) {
			int y = (int)ds->y[i];
			printf("[%d] ", y);
		}	else {
			printf("[%f] ", ds->y[i]);
		}
		ksvm_vector_print(ds->x[i]);
	}
}


//---------------------------------------------------------------------
// problem load
//---------------------------------------------------------------------
ksvm_problem_t* ksvm_problem_load(const char *filename, int right)
{
	ksvm_problem_t *pb = NULL; 
	char *line = NULL;
	int size = 0;
	int capacity = 1024;
	FILE *fp;
	int eof = 0;
	fp = fopen(filename, "r");
	if (fp == NULL) return NULL;
	pb = ksvm_problem_new();
	line = ksvm_new(char, capacity + 1);
	while (eof == 0) {
		// read line
		for (size = 0; ;) {
			int ch = fgetc(fp);
			if (ch < 0) {
				line[size++] = 0;
				eof = 1;
				break;
			}
			else if (ch == '\n') {
				line[size++] = 0;
				break;
			}
			else {
				if (size >= capacity) {
					capacity = capacity * 2;
					char *nt = ksvm_new(char, capacity + 1);
					memcpy(nt, line, size + 1);
					ksvm_free(line);
					line = nt;
				}
				line[size++] = (char)ch;
			}
		}
		ksvm_strip(line);
		if (line[0] == 0) 
			continue;
		if (right == 0) {
			char buffer[65];
			int len = 0;
			double y;
			ksvm_vector_t *vec;
			for (len = 0; len < 64 && line[len]; ) {
				if (isspace(line[len])) {
					buffer[len] = 0;
					break;
				}	else {
					buffer[len] = line[len];
					len++;
				}
			}
			y = atof(buffer);
			vec = ksvm_vector_load(line + len + 1);
			assert(vec);
			ksvm_problem_push(pb, vec, y);
		}
		else {
			ksvm_vector_t *vec = ksvm_vector_load(line);
			double y;
			assert(vec);
			assert(vec->size);
			y = vec->value[vec->size - 1];
			vec->size -= 1;
			ksvm_problem_push(pb, vec, y);
		}
	}
	ksvm_free(line);
	fclose(fp);
	return pb;
}


//---------------------------------------------------------------------
// problem group, max group is 16
//---------------------------------------------------------------------
#ifndef KSVM_MAX_CLASS
#define KSVM_MAX_CLASS    16
#endif

int ksvm_problem_group(const ksvm_problem_t *pb, int *labels, int *counts)
{
	int max_class = KSVM_MAX_CLASS;
	int count = 0;
	int i, j;
	for (i = 0; i < pb->nrows; i++) {
		int label = (int)pb->y[i];
		for (j = 0; j < count; j++) {
			if (labels[j] == label) break;
		}
		if (j < count) {
			counts[j]++;
		}
		else if (count < max_class) {
			labels[count] = label;
			counts[count] = 1;
			count++;
		}
	}
	return count;
}


//---------------------------------------------------------------------
// kernel init
//---------------------------------------------------------------------
void ksvm_kernel_init(ksvm_kernel_t *kernel, const ksvm_parameter_t *parameter)
{
	kernel->kernel_type = parameter->kernel_type;
	kernel->degree = parameter->degree;
	kernel->gamma = parameter->gamma;
	kernel->coef0 = parameter->coef0;
	kernel->pb = NULL;
	kernel->x = NULL;
	kernel->closure = NULL;
	kernel->func = NULL;
	kernel->square = NULL;
	kernel->nrows = 0;
	kernel->diag = NULL;
	kernel->cache = NULL;
	kernel->csize = 0;
	kernel->cache_on = 0;
	kernel->cache_miss = 0;
}


//---------------------------------------------------------------------
// kernel constructor
//---------------------------------------------------------------------
ksvm_kernel_t *ksvm_kernel_new(const ksvm_parameter_t *parameter, const ksvm_problem_t *pb)
{
	ksvm_kernel_t *kernel = ksvm_new(ksvm_kernel_t, 1);
	assert(kernel);
	kernel->kernel_type = parameter->kernel_type;
	kernel->degree = parameter->degree;
	kernel->gamma = parameter->gamma;
	kernel->coef0 = parameter->coef0;
	kernel->pb = pb;
	kernel->x = NULL;
	kernel->closure = NULL;
	kernel->func = NULL;
	kernel->square = NULL;
	kernel->nrows = 0;
	kernel->diag = NULL;
	kernel->cache = NULL;
	kernel->csize = 0;
	kernel->cache_on = 0;
	kernel->cache_miss = 0;
	if (kernel->pb && pb->nrows > 0) {
		int i, c;
		size_t cs;
		double limit = (parameter->cache_size <= 0)? 40 : parameter->cache_size;
		limit = ksvm_max(parameter->cache_size, 8) * 1024 * 1024;
		kernel->x = (const ksvm_vector_t**)(pb->x);
		kernel->nrows = pb->nrows;
		kernel->square = ksvm_new(double, pb->nrows);
		kernel->diag = ksvm_new(double, pb->nrows);
		assert(kernel->square);
		assert(kernel->diag);
		for (i = 0; i < pb->nrows; i++) {
			kernel->square[i] = ksvm_vector_dot(pb->x[i], pb->x[i]);
		}
		for (c = 1, cs = sizeof(ksvm_kcache_t); cs <= (size_t)limit; c *= 2, cs *= 2);
		kernel->cache = ksvm_new(ksvm_kcache_t, c);
		assert(kernel->cache);
		kernel->csize = c;
		kernel->cmask = c - 1;
		for (i = 0; i < c; i++) {
			kernel->cache[i].i = -1;
			kernel->cache[i].j = -1;
			kernel->cache[i].value = 0.0;
		}
		if (kernel->kernel_type != KSVM_KERNEL_USER) {
			for (i = 0; i < pb->nrows; i++) {
				kernel->diag[i] = ksvm_kernel_inner(kernel, i, i);
			}
		}	else {
			for (i = 0; i < pb->nrows; i++) {
				kernel->diag[i] = 0.0;
			}
		}
	}
	return kernel;
}


//---------------------------------------------------------------------
// dispose
//---------------------------------------------------------------------
void ksvm_kernel_free(ksvm_kernel_t *kernel)
{
	if (kernel->square) {
		ksvm_free(kernel->square);
		kernel->square = NULL;
	}
	if (kernel->diag) {
		ksvm_free(kernel->diag);
		kernel->diag = NULL;
	}
	if (kernel->cache) {
		ksvm_free(kernel->cache);
		kernel->cache = NULL;
	}
	kernel->func = NULL;
	kernel->closure = NULL;
	kernel->x = NULL;
	kernel->pb = NULL;
	kernel->cmask = 0;
	kernel->csize = 0;
	ksvm_free(kernel);
}


//---------------------------------------------------------------------
// install user function
//---------------------------------------------------------------------
void ksvm_kernel_install(ksvm_kernel_t *kernel, void *func, void *closure)
{
	typedef double (*ksvm_kernel_func_t)(void *, const ksvm_vector_t *, const ksvm_vector_t*);
	assert(kernel->kernel_type == KSVM_KERNEL_USER);
	kernel->func = (ksvm_kernel_func_t)func;
	kernel->closure = closure;
	if (func && kernel->nrows > 0) {
		int i;
		for (i = 0; i < kernel->nrows; i++) {
			kernel->diag[i] = ksvm_kernel_inner(kernel, i, i);
		}	
	}
}


//---------------------------------------------------------------------
// calculate
//---------------------------------------------------------------------
double ksvm_kernel_call(const ksvm_kernel_t *kernel, const ksvm_vector_t *x, const ksvm_vector_t *y)
{
	double m, n;
	switch (kernel->kernel_type)
	{
	case KSVM_KERNEL_LINEAR:
		return ksvm_vector_dot(x, y);
	case KSVM_KERNEL_POLY:
		return ksvm_powi(kernel->gamma * ksvm_vector_dot(x, y) + kernel->coef0, kernel->degree);
	case KSVM_KERNEL_RBF:
		m = ksvm_vector_dot(x, x);
		n = ksvm_vector_dot(y, y);
		return exp((-kernel->gamma) * (m + n - 2 * ksvm_vector_dot(x, y)));
	case KSVM_KERNEL_SIGMOID:
		return tanh(kernel->gamma * ksvm_vector_dot(x, y) + kernel->coef0);
	case KSVM_KERNEL_PRECOMPUTED:
		return x->value[y->uid];
	case KSVM_KERNEL_USER:
		assert(kernel->func);
		return kernel->func(kernel->closure, x, y);
	}
	return 0.0;
}


//---------------------------------------------------------------------
// calculate inner x
//---------------------------------------------------------------------
double ksvm_kernel_inner(const ksvm_kernel_t *kernel, int xi, int xj)
{
	double m, n, p;
	assert(kernel->x);
	switch (kernel->kernel_type)
	{
	case KSVM_KERNEL_LINEAR:
		return ksvm_vector_dot(kernel->x[xi], kernel->x[xj]);
	case KSVM_KERNEL_POLY:
		p = ksvm_vector_dot(kernel->x[xi], kernel->x[xj]);
		return ksvm_powi(kernel->gamma * p + kernel->coef0, kernel->degree);
	case KSVM_KERNEL_RBF:
		m = kernel->square[xi];
		n = kernel->square[xj];
		p = ksvm_vector_dot(kernel->x[xi], kernel->x[xj]);
		return exp((-kernel->gamma) * (m + n - 2 * p));
	case KSVM_KERNEL_SIGMOID:
		p = ksvm_vector_dot(kernel->x[xi], kernel->x[xj]);
		return tanh(kernel->gamma * p + kernel->coef0);
	case KSVM_KERNEL_PRECOMPUTED:
		return kernel->x[xi]->value[kernel->x[xj]->uid];
	case KSVM_KERNEL_USER:
		assert(kernel->func);
		return kernel->func(kernel->closure, kernel->x[xi], kernel->x[xj]);
	}
	return 0.0;
}


//---------------------------------------------------------------------
// query result from cache (if on) or recaculate it
//---------------------------------------------------------------------
double ksvm_kernel_query(ksvm_kernel_t *kernel, int xi, int xj)
{
	long long hash;
	unsigned int entry;
	ksvm_kcache_t *cache;
	if (xi == xj) {
		return kernel->diag[xi];
	}
	if (xi > xj) {
		if (kernel->kernel_type != KSVM_KERNEL_PRECOMPUTED && 
			kernel->kernel_type != KSVM_KERNEL_USER) {
			int t = xi;
			xi = xj;
			xj = t;
		}
	}
#if 1
	hash = ((long long)xi * kernel->nrows) + xj;
#else
	hash = (((long long)xi) * 0x3504f333) ^ (((long long)xj) * 0xf1bbcdcb);
	hash = ((unsigned int)hash) * 741103597;
#endif
	entry = (unsigned int)(hash & kernel->cmask);
	cache = &(kernel->cache[entry]);
	if (cache->i == xi && cache->j == xj) {
		kernel->cache_on++;
		return cache->value;
	}
	cache->i = xi;
	cache->j = xj;
	cache->value = ksvm_kernel_inner(kernel, xi, xj);
	kernel->cache_miss++;
	return cache->value;
}


//---------------------------------------------------------------------
// create solver
//---------------------------------------------------------------------
ksvm_solver_t* ksvm_solver_new(const ksvm_parameter_t *param, ksvm_kernel_t *kernel)
{
	ksvm_solver_t *solver = ksvm_new(ksvm_solver_t, 1);
	assert(solver);
	solver->nrows = 0;
	solver->kernel = kernel;
	solver->label_n = +1;
	solver->label_p = -1;
	solver->Cn = param->C;
	solver->Cp = param->C;
	solver->nrows = 0;
	solver->active_size = 0;
	solver->b = 0.0;
	solver->eps = param->eps;
	solver->index = NULL;
	solver->y = NULL;
	solver->svn = 0;
	solver->svx = NULL;
	solver->svy = NULL;
	solver->sva = NULL;
	solver->alpha = NULL;
	solver->G = NULL;
	return solver;
}


//---------------------------------------------------------------------
// free a solver
//---------------------------------------------------------------------
void ksvm_solver_free(ksvm_solver_t *solver)
{
	int i;
	assert(solver);
	if (solver->svx) {
		for (i = 0; i < solver->svn; i++) {
			if (solver->svx[i]) {
				ksvm_vector_free(solver->svx[i]);
				solver->svx[i] = NULL;
			}
		}
		ksvm_free(solver->svx);
		solver->svx = NULL;
	}
	if (solver->svy) {
		ksvm_free(solver->svy);
		solver->svy = NULL;
	}
	if (solver->sva) {
		ksvm_free(solver->sva);
		solver->sva = NULL;
	}
	if (solver->alpha) {
		ksvm_free(solver->alpha);
		solver->alpha = NULL;
	}
	if (solver->index) {
		ksvm_free(solver->index);
		solver->index = NULL;
	}
	if (solver->y) {
		ksvm_free(solver->y);
		solver->y = NULL;
	}
	if (solver->G) {
		ksvm_free(solver->G);
		solver->G = NULL;
	}
	solver->nrows = 0;
	solver->active_size = 0;
	ksvm_free(solver);
}


//---------------------------------------------------------------------
// resize
//---------------------------------------------------------------------
int ksvm_solver_resize(ksvm_solver_t *solver, int newsize)
{
	int osize = solver->nrows;
	int i;
	if (newsize <= 0) {
		if (solver->svx) {
			for (i = 0; i < solver->svn; i++) {
				if (solver->svx[i]) {
					ksvm_vector_free(solver->svx[i]);
					solver->svx[i] = NULL;
				}
			}
			ksvm_free(solver->svx);
		}
		solver->svx = NULL;
	}
	else if (solver->svx == NULL) {
		solver->svx = ksvm_new(ksvm_vector_t*, newsize);
		assert(solver->svx);
		for (i = 0; i < newsize; i++) {
			solver->svx[i] = NULL;
		}
	}
	else {
		ksvm_vector_t **saved = solver->svx;
		solver->svx = ksvm_new(ksvm_vector_t*, newsize);
		assert(solver->svx);
		for (i = 0; i < newsize; i++) {
			solver->svx[i] = (i < osize)? saved[i] : NULL;
		}
		for (; i < osize; i++) {
			if (saved[i]) {
				ksvm_vector_free(saved[i]);
				saved[i] = NULL;
			}
		}
		ksvm_free(saved);
	}
	solver->index = (int*)ksvm_realloc(solver->index, sizeof(int), osize, newsize);
	solver->y = (double*)ksvm_realloc(solver->y, sizeof(double), osize, newsize);
	solver->alpha = (double*)ksvm_realloc(solver->alpha, sizeof(double), osize, newsize);
	solver->G = (double*)ksvm_realloc(solver->G, sizeof(double), osize, newsize);
	solver->svy = (double*)ksvm_realloc(solver->svy, sizeof(double), osize, newsize);
	solver->sva = (double*)ksvm_realloc(solver->sva, sizeof(double), osize, newsize);
	solver->nrows = newsize;
	solver->active_size = ksvm_min(solver->active_size, solver->nrows);
	solver->svn = ksvm_min(solver->svn, solver->nrows);
	return 0;
}


//---------------------------------------------------------------------
// swap
//---------------------------------------------------------------------
int ksvm_solver_swap(ksvm_solver_t *solver, int i, int j)
{
	ksvm_i_swap(solver->index + i, solver->index + j);
	ksvm_f_swap(solver->y + i, solver->y + j);
	ksvm_f_swap(solver->alpha + i, solver->alpha + j);
	ksvm_f_swap(solver->G + i, solver->G + j);
	return 0;
}


//---------------------------------------------------------------------
// init svc
//---------------------------------------------------------------------
int ksvm_solver_init_svc(ksvm_solver_t *solver, const ksvm_parameter_t *param, int label_p, int label_n)
{
	const ksvm_problem_t *pb = solver->kernel->pb;
	int count_p = 0, count_n = 0, i, nrows, index;
	for (i = 0; i < pb->nrows; i++) {
		int label = (int)pb->y[i];
		if (label == label_p) count_p++;
		else if (label == label_n) count_n++;
	}
	nrows = count_p + count_n;
	if (nrows <= 0) 
		return -1;
	ksvm_solver_resize(solver, nrows);
	solver->active_size = solver->nrows;
	solver->label_p = label_p;
	solver->label_n = label_n;
	solver->b = 0;
	for (i = 0, index = 0; i < pb->nrows; i++) {
		int label = (int)pb->y[i];
		if (index < nrows) {
			if (label == label_p) {
				solver->index[index] = i;
				solver->y[index] = +1.0;
				index++;
			}
			else if (label == label_n) {
				solver->index[index] = i;
				solver->y[index] = -1.0;
				index++;
			}
		}	
	}
	for (i = 0; i < nrows; i++) {
		solver->alpha[i] = 0;
		solver->G[i] = 0;
	}
	assert(index == nrows);
	solver->Cp = param->C;
	solver->Cn = param->C;
	if (param->weight && param->weight_label) {
		for (i = 0; i < param->nweights; i++) {
			int label = param->weight_label[i];
			if (label == label_p) {
				solver->Cp = param->C * param->weight[i];
			}
			else if (label == label_n) {
				solver->Cn = param->C * param->weight[i];
			}
		}
	}
	return 0;
}


//---------------------------------------------------------------------
// get kernel result
//---------------------------------------------------------------------
double ksvm_solver_kernel(ksvm_solver_t *solver, int xi, int xj)
{
	int row_i = solver->index[xi];
	int row_j = solver->index[xj];
	return ksvm_kernel_query(solver->kernel, row_i, row_j);
}


//---------------------------------------------------------------------
// calculate G(xi) = sum([ aj * yj * K(xj, xi) for j in N ])
//---------------------------------------------------------------------
double ksvm_solver_calculate_gx(ksvm_solver_t *solver, int i, int active_size)
{
	double sum = 0.0;
	double *alpha = solver->alpha;
	double *y = solver->y;
	int j;
	if (active_size < 0) active_size = solver->nrows;
	for (j = 0; j < active_size; j++) {
		double a = alpha[j];
		double k;
		if (a == 0.0) continue;
		k = ksvm_solver_kernel(solver, j, i);
		sum += a * y[j] * k;
	}
	solver->G[i] = sum;
	return sum;
}


//---------------------------------------------------------------------
// inline functions
//---------------------------------------------------------------------
static inline double ksvm_solver_ci(const ksvm_solver_t *solver, int i)
{
	return (solver->y[i] > 0)? solver->Cp : solver->Cn;
}


//---------------------------------------------------------------------
// inner loop
//---------------------------------------------------------------------
#define KSVM_INF HUGE_VAL
#define KSVM_TAU 1e-12
#define KSVM_LTM 1e-8

int ksvm_solver_step(ksvm_solver_t *solver, int i, int j)
{
	double Gi, Gj, Ei, Ej, Ci, Cj, H, L, delta;
	double *alpha, *y, aj;
	double Kii, Kij, Kjj, b, b1, b2, diff;
	double eta, old_alpha_i, old_alpha_j;
	double delta_alpha_i;
	double delta_alpha_j;
	int k;

	if (i == j || i < 0 || j < 0) {
		// printf("exit1\n");
		return 0;
	}

#if 1
	Gi = solver->G[i];
	Gj = solver->G[j];
#else
	Gi = ksvm_solver_calculate_gx(solver, i, solver->nrows);
	Gj = ksvm_solver_calculate_gx(solver, j, solver->nrows);
#endif
	Ei = (Gi + solver->b) - solver->y[i];
	Ej = (Gj + solver->b) - solver->y[j];
	Ci = ksvm_solver_ci(solver, i);
	Cj = ksvm_solver_ci(solver, j);

	alpha = solver->alpha;
	y = solver->y;
	old_alpha_i = alpha[i];
	old_alpha_j = alpha[j];

	if (y[i] != y[j]) {
		L = ksvm_f_max(0, old_alpha_j - old_alpha_i);
		H = ksvm_f_min(Cj, Cj + old_alpha_j - old_alpha_i);
	}	else {
		L = ksvm_f_max(0, old_alpha_j + old_alpha_i - Cj);
		H = ksvm_f_min(Cj, old_alpha_j + old_alpha_i);
	}

	if (H <= L) {
		// printf("exit2\n");
		return 0;
	}

	Kii = ksvm_solver_kernel(solver, i, i);
	Kjj = ksvm_solver_kernel(solver, j, j);
	Kij = ksvm_solver_kernel(solver, i, j);

	eta = Kii + Kjj - 2 * Kij;

	if (eta <= 0) {
		// printf("fuck invalid eta\n");
		eta = KSVM_TAU;
	}

	delta = y[j] * (Ei - Ej) / eta;
	aj = alpha[j] + delta;
	aj = ksvm_f_mid(L, alpha[j] + delta, H);
	delta = aj - old_alpha_j;

	diff = (old_alpha_j * 2 + delta + solver->eps) * solver->eps;

	if (ksvm_f_abs(delta) < ksvm_f_abs(diff)) {
		// printf("exit3: %f %f\n", ksvm_f_abs(delta), ksvm_f_abs(diff));
		return 0;
	}

	alpha[j] = aj;
	alpha[i] += -delta * y[i] * y[j];
	delta_alpha_i = alpha[i] - old_alpha_i;
	delta_alpha_j = alpha[j] - old_alpha_j;

	for (k = 0; k < solver->active_size; k++) {
		double Kik = ksvm_solver_kernel(solver, i, k);
		double Kjk = ksvm_solver_kernel(solver, j, k);
		solver->G[k] += delta_alpha_i * Kik * y[i] + delta_alpha_j * Kjk * y[j];
	}

	b = solver->b;
	b1 = b - Ei - y[i] * delta_alpha_i * Kii - y[j] * delta_alpha_j * Kij;
	b2 = b - Ej - y[i] * delta_alpha_i * Kij - y[j] * delta_alpha_j * Kjj;

	if (alpha[i] > 0 && alpha[i] < Ci) {
		solver->b = b1;
	}
	else if (alpha[j] > 0 && alpha[j] < Cj) {
		solver->b = b2;
	}
	else {
		solver->b = (b1 + b2) * 0.5;
	}

	return 1;
}


//---------------------------------------------------------------------
// select
//---------------------------------------------------------------------
int ksvm_solver_getj(ksvm_solver_t *solver, int i)
{
	double Gi = solver->G[i];
	double Ei = (Gi + solver->b) - solver->y[i];
	double Kii = ksvm_solver_kernel(solver, i, i);
	double maxdelta = 0.0;
	int nrows = solver->active_size;
	int j, selected = -1;
	for (j = 0; j < nrows; j++) {
		double Gj, Ej, Kij, Kjj, eta, delta;
		if (i == j) continue;
		if (solver->alpha[j] == 0.0) continue;
		Gj = solver->G[j];
		Ej = (Gj + solver->b) - solver->y[j];
		Kjj = ksvm_solver_kernel(solver, j, j);
		Kij = ksvm_solver_kernel(solver, i, j);
		eta = Kii + Kjj - 2 * Kij;
		if (eta <= 0) eta = KSVM_TAU;
		delta = solver->y[j] * (Ei - Ej) / eta;
		delta = ksvm_f_abs(delta);
		if (delta > maxdelta) {
			maxdelta = delta;
			selected = j;
		}
	}
	if (selected > 0)
		return selected;
	while (1) {
		j = rand() % nrows;
		if (j != i) break;
	}
	return j;
}


//---------------------------------------------------------------------
// smo
//---------------------------------------------------------------------
int ksvm_solver_smo(ksvm_solver_t *solver, int max_iter)
{
	int nrows = solver->nrows;
	int iter = 0;
	int entire = 1;
	int changed = 0;
	int i, k = 0;
	max_iter = ksvm_max(max_iter, 10000);
	for (iter = 0; iter < max_iter; iter++) {
		if (entire == 0 && changed == 0) break;
		changed = 0;
		// entire = 1;
		if (entire) {
			for (i = 0; i < nrows; i++) {
				int j = ksvm_solver_getj(solver, i);
				// j = rand() % nrows;
				changed += ksvm_solver_step(solver, i, j);
			}
			// printf("entire: %d\n", changed);
		}
		else {
			for (i = 0; i < nrows; i++) {
				double Ci = ksvm_solver_ci(solver, i);
				if (solver->alpha[i] > 0 && solver->alpha[i] < Ci) {
					int j = ksvm_solver_getj(solver, i);
					changed += ksvm_solver_step(solver, i, j);
				}
			}
			// printf("partial: %d\n", changed);
		}
		if (entire) {
			entire = 0;
		}
		else if (changed == 0) {
			entire = 1;
		}
	}
	for (i = 0; i < nrows; i++) {
		if (solver->alpha[i] != 0) {
			int index = solver->index[i];
			solver->svx[k] = ksvm_vector_clone(solver->kernel->x[index]);
			solver->svy[k] = solver->y[i];
			solver->sva[k] = solver->alpha[i];
			k++;
		}
	}
	solver->svn = k;
	printf("svn: %d\n", k);
	printf("b: %f\n", solver->b);
	printf("active_size=%d\n", solver->active_size);
	return 0;
}


//---------------------------------------------------------------------
// ksvm_solver_train()
//---------------------------------------------------------------------
void ksvm_solver_train(ksvm_solver_t *solver)
{
	ksvm_solver_smo(solver, 100000);
}


//---------------------------------------------------------------------
// solver predict
//---------------------------------------------------------------------
int ksvm_solver_predict(const ksvm_solver_t *solver, const ksvm_vector_t *x)
{
	const ksvm_kernel_t *kernel = solver->kernel;
	double sum = solver->b;
	int i;
	for (i = 0; i < solver->svn; i++) {
		double k = ksvm_kernel_call(kernel, x, solver->svx[i]);
		sum += solver->sva[i] * solver->svy[i] * k;
	}
	return (sum > 0)? 1 : (-1);
}


//---------------------------------------------------------------------
// ksvm
//---------------------------------------------------------------------
ksvm_model_t* ksvm_model_new(const ksvm_parameter_t *param)
{
	ksvm_model_t *model = ksvm_new(ksvm_model_t, 1);
	model->param = param[0];
	model->nclasses = 0;
	model->size = 0;
	model->labels = NULL;
	model->svp = NULL;
	model->svn = NULL;
	model->svc = NULL;
	model->svx = NULL;
	model->rho = NULL;
	return model;
}


//---------------------------------------------------------------------
// free model
//---------------------------------------------------------------------
void ksvm_model_free(ksvm_model_t *model)
{
	int i;
	assert(model);
	if (model->svx) {
		for (i = 0; i < model->size; i++) {
			if (model->svx[i]) {
				ksvm_vector_free(model->svx[i]);
			}
			model->svx[i] = NULL;
		}	
		ksvm_free(model->svx);
		model->svx = NULL;
	}
	if (model->svc) {
		ksvm_free(model->svc);
		model->svc = NULL;
	}
	if (model->svp) {
		ksvm_free(model->svp);
		model->svp = NULL;
	}
	if (model->svn) {
		ksvm_free(model->svn);
		model->svn = NULL;
	}
	if (model->rho) {
		ksvm_free(model->rho);
		model->rho = NULL;
	}
	if (model->labels) {
		ksvm_free(model->labels);
		model->labels = NULL;
	}
	model->size = 0;
	model->nclasses = 0;
	ksvm_free(model);
}



//---------------------------------------------------------------------
// model resize support vector numbers
//---------------------------------------------------------------------
int ksvm_model_resize(ksvm_model_t *model, int newsize)
{
	int i;
	if (newsize <= 0) {
		if (model->svx) {
			for (i = 0; i < model->size; i++) {
				if (model->svx[i]) {
					ksvm_vector_free(model->svx[i]);
					model->svx[i] = NULL;
				}
			}
			ksvm_free(model->svx);
			model->svx = NULL;
		}	
	}
	else if (model->svx == NULL) {
		model->svx = ksvm_new(ksvm_vector_t*, newsize);
		assert(model->svx);
		for (i = 0; i < newsize; i++) {
			model->svx[i] = NULL;
		}
	}
	else {
		ksvm_vector_t **saved = model->svx;
		model->svx = ksvm_new(ksvm_vector_t*, newsize);
		assert(model->svx);
		for (i = 0; i < newsize; i++) {
			model->svx[i] = (i < model->size)? saved[i] : NULL;
		}
		for (; i < model->size; i++) {
			if (saved[i]) {
				ksvm_vector_free(saved[i]);
				saved[i] = NULL;
			}
		}
		ksvm_free(saved);
	}
	model->svc = (double*)ksvm_realloc(model->svc, sizeof(double), model->size, newsize);
	model->size = newsize;
	return 0;
}


//---------------------------------------------------------------------
// init class number
//---------------------------------------------------------------------
int ksvm_model_init_class(ksvm_model_t *model, int nclasses)
{
	int old_pairs = model->nclasses * (model->nclasses - 1) / 2;
	int new_pairs = nclasses * (nclasses - 1) / 2;
	model->labels = (int*)ksvm_realloc(model->labels, sizeof(int), model->nclasses, nclasses);
	model->svp = (int*)ksvm_realloc(model->svp, sizeof(int), old_pairs, new_pairs);
	model->svn = (int*)ksvm_realloc(model->svn, sizeof(int), old_pairs, new_pairs);
	model->rho = (double*)ksvm_realloc(model->rho, sizeof(double), old_pairs, new_pairs);
	model->nclasses = nclasses;
	return 0;
}


//---------------------------------------------------------------------
// train
//---------------------------------------------------------------------
int ksvm_model_train(ksvm_model_t *model, const ksvm_problem_t *pb)
{
	int labels[KSVM_MAX_CLASS];
	int counts[KSVM_MAX_CLASS];
	int nclasses, i, j, k;
	int x = 0, p = 0;
	ksvm_kernel_t *kernel;
	nclasses = ksvm_problem_group(pb, labels, counts);
	if (nclasses <= 1) return -1;
	ksvm_model_init_class(model, nclasses);
	assert(model->labels);
	kernel = ksvm_kernel_new(&model->param, pb);
	assert(kernel);
	for (i = 0; i < nclasses; i++) {
		model->labels[i] = labels[i];
	}
	for (i = 0; i < nclasses; i++) {
		for (j = i + 1; j < nclasses; j++, p++) {
			int label_i = model->labels[i];
			int label_j = model->labels[j];
			ksvm_solver_t *solver = ksvm_solver_new(&model->param, kernel);
			ksvm_solver_init_svc(solver, &model->param, label_i, label_j);
			ksvm_solver_train(solver);
			model->svp[p] = x;
			model->svn[p] = solver->svn;
			ksvm_model_resize(model, model->size + solver->svn);
			for (k = 0; k < solver->svn; x++, k++) {
				double alpha = solver->sva[k];
				double y = solver->svy[k];
				model->svx[x] = ksvm_vector_clone(solver->svx[k]);
				model->svc[x] = alpha * y;
			}
			model->rho[p] = solver->b;
			ksvm_solver_free(solver);
		}
	}
	return 0;
}



//---------------------------------------------------------------------
// predict values
//---------------------------------------------------------------------
double ksvm_model_predict(const ksvm_model_t *model, const ksvm_vector_t *x, double *dec_values)
{
	ksvm_kernel_t kernel;
	int i, j, k, p = 0;
	int votes[KSVM_MAX_CLASS];
	if (model->nclasses < 1) {
		return 0;
	}
	ksvm_kernel_init(&kernel, &model->param);
	for (i = 0; i < KSVM_MAX_CLASS; i++)
		votes[i] = 0;
	for (i = 0; i < model->nclasses; i++) {
		for (j = i + 1; j < model->nclasses; j++, p++) {
			double sum = model->rho[p];	
			int svn = model->svn[p];
			int start = model->svp[p];
			for (k = 0; k < svn; k++, start++) {
				const ksvm_vector_t *svx = model->svx[start];
				double kk = ksvm_kernel_call(&kernel, x, svx);
				sum += model->svc[start] * kk;
			}
			if (dec_values) {
				dec_values[p] = sum;
			}
			if (sum > 0) votes[i]++;
			else votes[j]++;
		}
	}
	for (i = 0, k = 0; i < model->nclasses; i++) {
		if (votes[i] > votes[k])
			k = i;
	}
	return model->labels[k];
}


