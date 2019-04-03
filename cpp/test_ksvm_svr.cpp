#include "ksvm.c"

void init_param(ksvm_parameter_t *param, double C, double gamma)
{
	memset(param, 0, sizeof(ksvm_parameter_t));
	param->svm_type = KSVM_TYPE_SVR;
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
	param->p = 0.125;
}


void test1()
{
	ksvm_parameter_t param;
	ksvm_problem_t *pb = ksvm_problem_load("../data/battery_data.txt", 1);
	init_param(&param, 1024, 1.3);
	ksvm_model_t *model;
	model = ksvm_model_new(&param);
	ksvm_model_train(model, pb);

	double error = 0.0;
	for (int i = 0; i < pb->nrows; i++) {
		double h = ksvm_model_predict(model, pb->x[i], NULL);	
		double y = pb->y[i];
		printf("%f -> %f, delta %f\n", h, y, ksvm_f_abs(h - y));
		error += ksvm_f_abs(h - y) * ksvm_f_abs(h - y);
	}
	printf("error=%f\n", error);
}

int main(void)
{
	test1();
	return 0;
}


