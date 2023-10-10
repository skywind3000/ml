#include "ksvm.c"

void init_param(ksvm_parameter_t *param, double C, double gamma)
{
	memset(param, 0, sizeof(ksvm_parameter_t));
	param->svm_type = KSVM_TYPE_SVC;
	param->kernel_type = KSVM_KERNEL_RBF;
	param->degree = 1;
	param->gamma = 1.0 / (gamma * gamma);
	param->coef0 = 0;
	param->C = C;
	param->eps = 0.001;
	param->nweights = 0;
	param->weight = NULL;
	param->weight_label = NULL;
	param->cache_size = 40;
}

void test1()
{
	ksvm_problem_t *pb = ksvm_problem_load("../data/testSetRBF.txt", 1);
	ksvm_problem_t *pt = ksvm_problem_load("../data/testSetRBF2.txt", 1);
	ksvm_parameter_t param;
	init_param(&param, 200, 1.3);
	ksvm_model_t *model = ksvm_model_new(&param);
	ksvm_model_train(model, pb);
	int e = 0;
	for (int i = 0; i < pt->nrows; i++) {
		int label = ksvm_model_predict(model, pt->x[i], NULL);
		if (label != pt->y[i]) e++;
	}
	printf("test error rate: %f\n", ((float)e) / pt->nrows);
	// printf("HUGE: %f\n", HUGE_VAL);
}

int main(void)
{
	test1();
	return 0;
}



