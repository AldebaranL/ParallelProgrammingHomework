#include "ANN_parallel.h"

ANN_parallel::ANN_parallel(int* _num_each_layer, int _num_epoch, int _batch_size, int _num_layers, float _study_rate)
{
	num_layers = _num_layers;
	study_rate = _study_rate;
	num_each_layer = new int[_num_layers + 2];
	num_each_layer[0] = _num_each_layer[0];
	for (int i = 1; i <= num_layers + 1; i++)
	{
		num_each_layer[i] = _num_each_layer[i];
		layers.push_back(new Layer(num_each_layer[i - 1], num_each_layer[i], _tanh));
	}
	//display();
	num_epoch = _num_epoch;
	batch_size = _batch_size;
}
ANN_parallel::~ANN_parallel()
{
	//printf ("begin free ANN_parallel");
	delete[]num_each_layer;
	for (int i = 0; i < layers.size(); i++) delete layers[i];
	// printf ("free ANN_parallel\n");
}
void ANN_parallel::shuffle(const int num_sample, float** _trainMat, float** _labelMat)
{
	//init
	int* shuffle_index = new int[num_sample];
	float** trainMat_old = new float* [num_sample];
	float** labelMat_old = new float* [num_sample];
	for (int i = 0; i < num_sample; i++)
	{
		trainMat_old[i] = new float[num_each_layer[0]];
		labelMat_old[i] = new float[num_each_layer[num_layers + 1]];
	}
	for (int i = 0; i < num_sample; i++)
	{
		for (int j = 0; j < num_each_layer[0]; j++)
			trainMat_old[i][j] = _trainMat[i][j];
		for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
			labelMat_old[i][j] = _labelMat[i][j];
	}

	//shuffle
	for (int i = 0; i < num_sample; i++) shuffle_index[i] = i;
	random_shuffle(shuffle_index, shuffle_index + num_sample);
	for (int i = 0; i < num_sample; i++)
	{
		for (int j = 0; j < num_each_layer[0]; j++) _trainMat[i][j] = trainMat_old[shuffle_index[i]][j];
		for (int j = 0; j < num_each_layer[num_layers + 1]; j++) _labelMat[i][j] = labelMat_old[shuffle_index[i]][j];
	}

	//delete
	for (int i = 0; i < num_sample; i++)
	{
		delete[]trainMat_old[i];
		delete[]labelMat_old[i];
	}
	delete[]trainMat_old;
	delete[]labelMat_old;
	delete[] shuffle_index;

	printf("finish shuffle\n");
}
void ANN_parallel::get_results(const int num_sample, float** _trainMat, float** _labelMat) {
	float tp = 0, fp = 0, tn = 0, fn = 0;
	float* Y;
	for (int index = 0; index < num_sample; index++) {
		this->predict(_trainMat[index]);
		Y = layers[num_layers]->output_nodes;
		float maxn = -1, maxi = 0;
		for (int j = 0; j < num_each_layer[num_layers + 1]; j++) {
			if (Y[j] > maxn) {
				maxn = Y[j];
				maxi = j;
			}
		}
		if (_labelMat[index][(int)maxi] == 1) {
			tp++;
		}
		else {
			fp++;
			cout << maxn << maxi;
			this->show_predictions(_trainMat[index]);
		}
	}
	cout << "tp:" << tp << endl;
	cout << "fp:" << fp << endl;
	cout << "accuracy:" << tp / (tp + fp) << endl;
}

void ANN_parallel::train(const int num_sample, float** _trainMat, float** _labelMat)
{
	std::printf("begin training\n");
	float thre = 1e-2;
	long long head, tail, freq;// timers
	long long head1, tail1;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time

	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		//isNotConver_(num_sample, _trainMat, _labelMat, thre);
		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
			for (int i = num_layers; i >= 0; i--)
				for (int j = 0; j < num_each_layer[i + 1]; j++) {
					for (int k = 0; k < num_each_layer[i]; k++) {
						layers[i]->weights_delta[j][k] = 0;
					}
					layers[i]->bias_delta[j] = 0;
				}
			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;
			for (int index = batch_i * batch_size; index < batch_i * batch_size + t_batch_size; index++) {
				QueryPerformanceCounter((LARGE_INTEGER*)&head1); // start time
				//前向传播
				float output_node;
				for (int i = 0; i < num_each_layer[1]; i++)
				{
					output_node = 0.0;
					for (int j = 0; j < num_each_layer[0]; j++)
					{
						output_node += layers[0]->weights[i][j] * _trainMat[index][j];
					}
					output_node += layers[0]->bias[i];
					layers[0]->output_nodes[i] = layers[0]->activation_function(output_node);//输出-1到1
				}
				for (int i_layer = 1; i_layer <= num_layers; i_layer++)
				{
					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
					{
						output_node = 0.0;
						for (int j = 0; j < num_each_layer[i_layer]; j++)
						{
							output_node += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
						}
						output_node += layers[i_layer]->bias[i];
						layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(output_node);
					}
				}
				//计算delta
				float* temp_bias_delta = new float[MAX_NUM_LAYER];
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数
					temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
					layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
					for (int k = 0; k < num_each_layer[num_layers]; k++)
						layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
				}
				//计算每层的delta,行访问优化
				for (int i = num_layers - 1; i >= 0; i--)
				{
					float* error = new float[num_each_layer[i + 1]];

					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
					for (int k = 0; k < num_each_layer[i + 2]; k++)
					{
						for (int j = 0; j < num_each_layer[i + 1]; j++)
						{
							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
						}
					}
					for (int j = 0; j < num_each_layer[i + 1]; j++)
					{
						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
						layers[i]->bias_delta[j] += temp_bias_delta[j];
						for (int k = 0; k < num_each_layer[i]; k++)
							if (i == 0) layers[0]->weights_delta[j][k] += temp_bias_delta[j] * _trainMat[index][k];
							else layers[i]->weights_delta[j][k] += temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
					}
					delete[]error;
				}
				QueryPerformanceCounter((LARGE_INTEGER*)&tail1); // start time
				//std::cout << "one index:" <<index << (tail1 - head1) * 1.0 / freq << "s" << endl;
				delete[]temp_bias_delta;
			}

			//反向传播，weights和bias更新
			for (int i = 0; i <= num_layers; i++)
			{
				for (int k = 0; k < num_each_layer[i + 1]; k++)
				{
					for (int j = 0; j < num_each_layer[i]; j++)
					{
						layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
					}
					layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
				}
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
	std::cout << "seriel_all:" << (tail - head) * 1.0 / freq << "s" << endl;
	std::cout << "one index" << ': ' << (tail1 - head1) * 1.0 / freq << "s" << endl;
	//std::printf("finish training\n");
}

void ANN_parallel::train_openMP_SIMD(const int num_sample, float** _trainMat, float** _labelMat)
{
	std::printf("begin training\n");
	float thre = 1e-2;
	long long head, tail, freq;// timers
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time

	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		//isNotConver_(num_sample, _trainMat, _labelMat, thre);
		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
#pragma omp parallel num_threads(NUM_THREADS)
		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
#pragma omp for
			for (int i = num_layers; i >= 0; i--) {
				for (int j = 0; j < num_each_layer[i + 1]; j++) {
					memset(layers[i]->weights_delta[j], 0, num_each_layer[i]);
				}
				memset(layers[i]->bias_delta, 0, num_each_layer[i + 1]);
			}
			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;
			for (int index = batch_i * batch_size; index < batch_i * batch_size + t_batch_size; index++) {
				//前向传播
#pragma omp for
				for (int i = 0; i < num_each_layer[1]; i++)
				{
					layers[0]->output_nodes[i] = 0.0;
					//printf("%d,%d ",i,class_p->num_each_layer[1]);
					int j = 0;
					__m128 ans = _mm_setzero_ps();
					for (; j + 4 < num_each_layer[0]; j += 4)
					{
						__m128 t1, t2;
						//将内部循环改为4位的向量运算
						t1 = _mm_loadu_ps(layers[0]->weights[i] + j);
						t2 = _mm_loadu_ps(_trainMat[index] + j);
						t1 = _mm_mul_ps(t1, t2);

						ans = _mm_add_ps(ans, t1);
					}
					// 4个局部和相加
					ans = _mm_hadd_ps(ans, ans);
					ans = _mm_hadd_ps(ans, ans);
					//标量存储
					float z;
					_mm_store_ss(&z, ans);
					if (j == 0)j += 4;
					for (j -= 4; j < num_each_layer[0]; j++) {
						z += layers[0]->weights[i][j] * _trainMat[index][j];
					}
					layers[0]->output_nodes[i] += z;
					layers[0]->output_nodes[i] += layers[0]->bias[i];
					layers[0]->output_nodes[i] = layers[0]->activation_function(layers[0]->output_nodes[i]);
				}
				for (int layers_i = 1; layers_i <= num_layers; layers_i++)
				{
#pragma omp for
					for (int i = 0; i < num_each_layer[layers_i + 1]; i++)
					{
						layers[layers_i]->output_nodes[i] = 0.0;
						int j = 0;
						__m128 ans = _mm_setzero_ps();
						for (; j + 4 < num_each_layer[layers_i]; j += 4)
						{
							__m128 t1, t2;
							//将内部循环改为4位的向量运算
							t1 = _mm_loadu_ps(layers[layers_i]->weights[i] + j);
							t2 = _mm_loadu_ps(layers[layers_i - 1]->output_nodes + j);
							t1 = _mm_mul_ps(t1, t2);

							ans = _mm_add_ps(ans, t1);
						}
						// 4个局部和相加
						ans = _mm_hadd_ps(ans, ans);
						ans = _mm_hadd_ps(ans, ans);
						//标量存储
						float z;
						_mm_store_ss(&z, ans);
						if (j == 0)j += 4;
						for (j -= 4; j < num_each_layer[layers_i]; j++) {
							z += layers[layers_i]->weights[i][j] * layers[layers_i - 1]->output_nodes[j];
						}
						layers[layers_i]->output_nodes[i] += z;
						layers[layers_i]->output_nodes[i] += layers[layers_i]->bias[i];
						layers[layers_i]->output_nodes[i] = layers[layers_i]->activation_function(layers[layers_i]->output_nodes[i]);
					}
				}

				float* temp_bias_delta = new float[MAX_NUM_LAYER];
#pragma omp for
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数
					temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
					layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
					int k = 0;
					__m128 t2 = _mm_set1_ps(temp_bias_delta[j]);
					for (; k + 4 < num_each_layer[num_layers]; k += 4)
					{
						__m128 t1, t3, product;
						t1 = _mm_loadu_ps(layers[num_layers]->weights_delta[j] + k);
						t3 = _mm_loadu_ps(layers[num_layers - 1]->output_nodes + k);
						//product=sr*t2*t3，向量对位相乘
						product = _mm_mul_ps(t3, t2);
						t1 = _mm_add_ps(t1, product);
						_mm_storeu_ps(layers[num_layers]->weights_delta[j] + k, t1);
					}
					if (k == 0)k += 4;
					for (k -= 4; k < num_each_layer[num_layers]; k++)
						layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
				}
				//计算每层的delta,行访问优化
				for (int i = num_layers - 1; i >= 0; i--)
				{
					float* error = new float[num_each_layer[i + 1]];

					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
#pragma omp for
					for (int k = 0; k < num_each_layer[i + 2]; k++)
					{
						int j = 0;
						__m128 t2 = _mm_set1_ps(temp_bias_delta[k]);
						for (; j + 4 < num_each_layer[i + 1]; j += 4)
						{
							__m128 t1, t3, product;
							t1 = _mm_loadu_ps(error + j);
							t3 = _mm_loadu_ps(&layers[i + 1]->weights[k][j]);
							product = _mm_mul_ps(t3, t2);
							t1 = _mm_add_ps(t1, product);
							_mm_storeu_ps(error + j, t1);
						}
						if (j == 0)j += 4;
						for (j -= 4; j < num_each_layer[i + 1]; j++) {
							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
						}
					}
#pragma omp for
					for (int j = 0; j < num_each_layer[i + 1]; j++)
					{
						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
						layers[i]->bias_delta[j] += temp_bias_delta[j];
						int k = 0;
						__m128 t2 = _mm_set1_ps(temp_bias_delta[j]);
						for (; k + 4 < num_each_layer[i]; k += 4)
						{
							__m128 t1, t3, product;
							t1 = _mm_loadu_ps(&layers[i]->weights_delta[j][k]);
							if (i == 0) t3 = _mm_loadu_ps(&_trainMat[index][k]);
							else t3 = _mm_loadu_ps(&layers[i - 1]->output_nodes[k]);
							product = _mm_mul_ps(t3, t2);
							t1 = _mm_add_ps(t1, product);
							_mm_storeu_ps(&layers[i]->weights_delta[j][k], t1);
						}
						if (k == 0)k += 4;
						for (k -= 4; k < num_each_layer[i]; k++)
							if (i == 0) layers[0]->weights_delta[j][k] += temp_bias_delta[j] * _trainMat[index][k];
							else layers[i]->weights_delta[j][k] += temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
					}
					delete[]error;
				}

				delete[]temp_bias_delta;
			}

			//反向传播，weights和bias更新
			for (int i = 0; i <= num_layers; i++)
			{
#pragma omp for
				for (int k = 0; k < num_each_layer[i + 1]; k++)
				{
					__m128 r = _mm_set1_ps(study_rate / t_batch_size);
					int j = 0;
					for (; j + 4 < num_each_layer[i]; j += 4) {
						__m128 t1, t3, product;
						t1 = _mm_loadu_ps(&layers[i]->weights[k][j]);
						t3 = _mm_loadu_ps(&layers[i]->weights_delta[k][j]);
						product = _mm_mul_ps(t3, r);
						t1 = _mm_sub_ps(t1, product);
						_mm_storeu_ps(&layers[i]->weights[k][j], t1);
					}
					if (j == 0)j += 4;
					for (j -= 4; j < num_each_layer[i]; j++) {
						layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
					}
					layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
				}
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
	std::cout << "SIMD+openMP:" << (tail - head) * 1.0 / freq << "s" << endl;
	//std::printf("finish training\n");
}
void ANN_parallel::train_SIMD(const int num_sample, float** _trainMat, float** _labelMat)
{
	std::printf("begin training\n");
	float thre = 1e-2;
	long long head, tail, freq;// timers
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time

	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		//isNotConver_(num_sample, _trainMat, _labelMat, thre);
		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
			for (int i = num_layers; i >= 0; i--){
				for (int j = 0; j < num_each_layer[i + 1]; j++) {
					memset(layers[i]->weights_delta[j], 0, num_each_layer[i]);
				}
				memset(layers[i]->bias_delta, 0, num_each_layer[i + 1]);
			}
			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;
			for (int index = batch_i * batch_size; index < batch_i * batch_size + t_batch_size; index++) {
				//前向传播
				for (int i = 0; i < num_each_layer[1]; i++)
				{
					layers[0]->output_nodes[i] = 0.0;
					//printf("%d,%d ",i,class_p->num_each_layer[1]);
					int j=0;
					__m128 ans = _mm_setzero_ps();
					for (; j+4 < num_each_layer[0]; j += 4)
					{
						__m128 t1, t2;
						//将内部循环改为4位的向量运算
						t1 = _mm_loadu_ps(layers[0]->weights[i] + j);
						t2 = _mm_loadu_ps(_trainMat[index] + j);
						t1 = _mm_mul_ps(t1, t2);

						ans = _mm_add_ps(ans, t1);
					}
					// 4个局部和相加
					ans = _mm_hadd_ps(ans, ans);
					ans = _mm_hadd_ps(ans, ans);
					//标量存储
					float z;
					_mm_store_ss(&z, ans);
					if (j == 0)j += 4;
					for (j-=4; j < num_each_layer[0]; j++) {
						z += layers[0]->weights[i][j] * _trainMat[index][j];
					}
					layers[0]->output_nodes[i] += z;
					layers[0]->output_nodes[i] += layers[0]->bias[i];
					layers[0]->output_nodes[i] = layers[0]->activation_function(layers[0]->output_nodes[i]);
				}
				for (int layers_i = 1; layers_i <= num_layers; layers_i++)
				{

					for (int i = 0; i < num_each_layer[layers_i + 1]; i++)
					{
						layers[layers_i]->output_nodes[i] = 0.0;
						int j = 0;
						__m128 ans = _mm_setzero_ps();
						for (; j + 4 < num_each_layer[layers_i]; j += 4)
						{
							__m128 t1, t2;
							//将内部循环改为4位的向量运算
							t1 = _mm_loadu_ps(layers[layers_i]->weights[i] + j);
							t2 = _mm_loadu_ps(layers[layers_i - 1]->output_nodes + j);
							t1 = _mm_mul_ps(t1, t2);

							ans = _mm_add_ps(ans, t1);
						}
						// 4个局部和相加
						ans = _mm_hadd_ps(ans, ans);
						ans = _mm_hadd_ps(ans, ans);
						//标量存储
						float z;
						_mm_store_ss(&z, ans);
						if (j == 0)j += 4;
						for (j-=4; j < num_each_layer[layers_i]; j++) {
							z += layers[layers_i]->weights[i][j] * layers[layers_i - 1]->output_nodes[j];
						}
						layers[layers_i]->output_nodes[i] += z;
						layers[layers_i]->output_nodes[i] += layers[layers_i]->bias[i];
						layers[layers_i]->output_nodes[i] = layers[layers_i]->activation_function(layers[layers_i]->output_nodes[i]);
					}
				}

				float* temp_bias_delta = new float[MAX_NUM_LAYER];
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数
					temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
					layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
					int k = 0;
					__m128 t2 = _mm_set1_ps(temp_bias_delta[j]);
					for (; k + 4 < num_each_layer[num_layers]; k += 4)
					{
						__m128 t1, t3, product;
						t1 = _mm_loadu_ps(layers[num_layers]->weights_delta[j]+k);
						t3 = _mm_loadu_ps(layers[num_layers - 1]->output_nodes + k);
						//product=sr*t2*t3，向量对位相乘
						product = _mm_mul_ps(t3, t2);
						t1 = _mm_add_ps(t1, product);
						_mm_storeu_ps(layers[num_layers]->weights_delta[j]+k, t1);
					}
					if (k == 0)k += 4;
					for (k-=4; k < num_each_layer[num_layers]; k++)
						layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
				}
				//计算每层的delta,行访问优化
				for (int i = num_layers - 1; i >= 0; i--)
				{
					float* error = new float[num_each_layer[i + 1]];

					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
					for (int k = 0; k < num_each_layer[i + 2]; k++)
					{
						int j = 0;
						__m128 t2 = _mm_set1_ps(temp_bias_delta[k]);
						for (; j + 4 < num_each_layer[i+1]; j += 4)
						{
							__m128 t1, t3, product;
							t1 = _mm_loadu_ps(error + j);
							t3 = _mm_loadu_ps(&layers[i + 1]->weights[k][j]);
							product = _mm_mul_ps(t3, t2);
							t1 = _mm_add_ps(t1, product);
							_mm_storeu_ps(error + j, t1);
						}
						if (j == 0)j += 4;
						for (j-=4; j < num_each_layer[i + 1]; j++){
							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
						}
					}
					for (int j = 0; j < num_each_layer[i + 1]; j++)
					{
						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
						layers[i]->bias_delta[j] += temp_bias_delta[j];
						int k = 0;
						__m128 t2 = _mm_set1_ps(temp_bias_delta[j]);
						for (; k+4 < num_each_layer[i]; k += 4)
						{
							__m128 t1, t3, product;
							t1 = _mm_loadu_ps(&layers[i]->weights_delta[j][k]);
							if (i == 0) t3 = _mm_loadu_ps(&_trainMat[index][k]);
							else t3 = _mm_loadu_ps(&layers[i - 1]->output_nodes[k]);
							product = _mm_mul_ps(t3, t2);
							t1 = _mm_add_ps(t1, product);
							_mm_storeu_ps(&layers[i]->weights_delta[j][k], t1);
						}
						if (k == 0)k += 4;
						for (k-=4; k < num_each_layer[i]; k++)
							if (i == 0) layers[0]->weights_delta[j][k] += temp_bias_delta[j] * _trainMat[index][k];
							else layers[i]->weights_delta[j][k] += temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
					}
					delete[]error;
				}

				delete[]temp_bias_delta;
			}

			//反向传播，weights和bias更新
			for (int i = 0; i <= num_layers; i++)
			{
				for (int k = 0; k < num_each_layer[i + 1]; k++)
				{
					__m128 r = _mm_set1_ps(study_rate / t_batch_size);
					int j = 0;
					for (; j + 4 < num_each_layer[i]; j += 4) {
						__m128 t1, t3, product;
						t1 = _mm_loadu_ps(&layers[i]->weights[k][j]);
						t3 = _mm_loadu_ps(&layers[i]->weights_delta[k][j]);
						product = _mm_mul_ps(t3, r);
						t1 = _mm_sub_ps(t1, product);
						_mm_storeu_ps(&layers[i]->weights[k][j], t1);
					}
					if (j == 0)j += 4;
					for (j-=4; j < num_each_layer[i]; j++){
						layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
					}
					layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
				}
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
	std::cout << "SIMD:" << (tail - head) * 1.0 / freq << "s" << endl;
	//std::printf("finish training\n");
}

void ANN_parallel::train_openMP(const int num_sample, float** _trainMat, float** _labelMat)
{
	std::printf("begin training\n");
	float thre = 1e-2;
	long long head, tail, freq;// timers
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time

	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		//isNotConver_(num_sample, _trainMat, _labelMat, thre);
		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
#pragma omp parallel num_threads(NUM_THREADS)
		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
			for (int i = num_layers; i >= 0; i--) {

#pragma omp for
				for (int j = 0; j < num_each_layer[i + 1]; j++) {
					memset(layers[i]->weights_delta[j], 0, num_each_layer[i]);
				}
				memset(layers[i]->bias_delta, 0, num_each_layer[i + 1]);
			}
			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;
			for (int index = batch_i * batch_size; index < batch_i * batch_size + t_batch_size; index++) {
				//前向传播
				float output_node;
#pragma omp for
				for (int i = 0; i < num_each_layer[1]; i++)
				{
					output_node = 0.0;
					for (int j = 0; j < num_each_layer[0]; j++)
					{
						output_node += layers[0]->weights[i][j] * _trainMat[index][j];
					}
					output_node += layers[0]->bias[i];
					layers[0]->output_nodes[i] = layers[0]->activation_function(output_node);//输出-1到1
				}
				for (int i_layer = 1; i_layer <= num_layers; i_layer++)
				{
#pragma omp for
					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
					{
						output_node = 0.0;
						for (int j = 0; j < num_each_layer[i_layer]; j++)
						{
							output_node += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
						}
						output_node += layers[i_layer]->bias[i];
						layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(output_node);
					}
				}
				//计算delta
				float* temp_bias_delta = new float[MAX_NUM_LAYER];
#pragma omp for
				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数
					temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
					layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
					for (int k = 0; k < num_each_layer[num_layers]; k++)
						layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
				}
				//计算每层的delta,行访问优化
				for (int i = num_layers - 1; i >= 0; i--)
				{
					float* error = new float[num_each_layer[i + 1]];

					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
					for (int k = 0; k < num_each_layer[i + 2]; k++)
					{
#pragma omp for
						for (int j = 0; j < num_each_layer[i + 1]; j++)
						{
							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
						}
					}
#pragma omp for
					for (int j = 0; j < num_each_layer[i + 1]; j++)
					{
						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
						layers[i]->bias_delta[j] += temp_bias_delta[j];
						for (int k = 0; k < num_each_layer[i]; k++)
							if (i == 0) layers[0]->weights_delta[j][k] += temp_bias_delta[j] * _trainMat[index][k];
							else layers[i]->weights_delta[j][k] += temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
					}
					delete[]error;
				}

				delete[]temp_bias_delta;
			}

			//反向传播，weights和bias更新
			for (int i = 0; i <= num_layers; i++)
			{
#pragma omp for
				for (int k = 0; k < num_each_layer[i + 1]; k++)
				{
					for (int j = 0; j < num_each_layer[i]; j++)
					{
						layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
					}
					layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
				}
			}
		
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
	std::cout << "openMP:" << (tail - head) * 1.0 / freq << "s" << endl;
	std::printf("finish training\n");
}
void ANN_parallel::train_MPI_openMP_SIMD(const int num_sample, float** _trainMat, float** _labelMat)
{
	std::printf("begin training\n");
	long long head, tail, freq;// timers
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time

	float thre = 1e-2;
	int myid, numprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		//if (myid == 0) isNotConver_(num_sample, _trainMat, _labelMat, thre);
		//fflush(stdout);
		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
			if (myid == 0) {
				for (int i = num_layers; i >= 0; i--) {
					for (int j = 0; j < num_each_layer[i + 1]; j++) {
						memset(layers[i]->weights_delta[j], 0, num_each_layer[i]);
					}
					memset(layers[i]->bias_delta, 0, num_each_layer[i + 1]);
				}
			}
			//cout << myid << "is waiting in bari1" << endl;
			//MPI_Barrier(MPI_COMM_WORLD);
			//cout << myid << "after bari1" << endl;
			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;

			int my_size = (t_batch_size + numprocs - 1) / numprocs;
			//
			for (int index = batch_i * batch_size + my_size * myid;
				index < min(batch_i * batch_size + t_batch_size, batch_i * batch_size + my_size * (myid + 1));
				index++) {
				//cout << myid << "index" <<index<< endl;
				//数据发送
				for (int i_layer = 0; i_layer <= num_layers; i_layer++) {
					MPI_Bcast(layers[i_layer]->bias, num_each_layer[i_layer + 1], MPI_FLOAT, 0, MPI_COMM_WORLD);
					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
						MPI_Bcast(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
				}
				//cout <<myid<< "begin predict" << endl;
								//前向传播
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
				for (int i = 0; i < num_each_layer[1]; i++)
				{
					layers[0]->output_nodes[i] = 0.0;
					//printf("%d,%d ",i,class_p->num_each_layer[1]);
					int j = 0;
					__m128 ans = _mm_setzero_ps();
					for (; j + 4 < num_each_layer[0]; j += 4)
					{
						__m128 t1, t2;
						//将内部循环改为4位的向量运算
						t1 = _mm_loadu_ps(layers[0]->weights[i] + j);
						t2 = _mm_loadu_ps(_trainMat[index] + j);
						t1 = _mm_mul_ps(t1, t2);

						ans = _mm_add_ps(ans, t1);
					}
					// 4个局部和相加
					ans = _mm_hadd_ps(ans, ans);
					ans = _mm_hadd_ps(ans, ans);
					//标量存储
					float z;
					_mm_store_ss(&z, ans);
					for (j -= 4; j < num_each_layer[0]; j++) {
						z += layers[0]->weights[i][j] * _trainMat[index][j];
					}
					layers[0]->output_nodes[i] += z;
					layers[0]->output_nodes[i] += layers[0]->bias[i];
					layers[0]->output_nodes[i] = layers[0]->activation_function(layers[0]->output_nodes[i]);
				}
				for (int layers_i = 1; layers_i <= num_layers; layers_i++)
				{
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
					for (int i = 0; i < num_each_layer[layers_i + 1]; i++)
					{
						layers[layers_i]->output_nodes[i] = 0.0;
						int j = 0;
						__m128 ans = _mm_setzero_ps();
						for (; j + 4 < num_each_layer[layers_i]; j += 4)
						{
							__m128 t1, t2;
							//将内部循环改为4位的向量运算
							t1 = _mm_loadu_ps(layers[layers_i]->weights[i] + j);
							t2 = _mm_loadu_ps(layers[layers_i - 1]->output_nodes + j);
							t1 = _mm_mul_ps(t1, t2);

							ans = _mm_add_ps(ans, t1);
						}
						// 4个局部和相加
						ans = _mm_hadd_ps(ans, ans);
						ans = _mm_hadd_ps(ans, ans);
						//标量存储
						float z;
						_mm_store_ss(&z, ans);
						for (j -= 4; j < num_each_layer[layers_i]; j++) {
							z += layers[layers_i]->weights[i][j] * layers[layers_i - 1]->output_nodes[j];
						}
						layers[layers_i]->output_nodes[i] += z;
						layers[layers_i]->output_nodes[i] += layers[layers_i]->bias[i];
						layers[layers_i]->output_nodes[i] = layers[layers_i]->activation_function(layers[layers_i]->output_nodes[i]);
					}
				}
				//cout <<myid<< "begin cal delta" << endl;
				//计算delta
				MPI_Status status;
				//cout << MAX_NUM_LAYER << endl;
				//for (int i = 0; i < num_layers + 2; i++)cout << num_each_layer[i] << ' ';
				//float* temp_bias_delta = new float[MAX_NUM_LAYER];
				float* temp_bias_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));
				//float* temp_weights_delta = new float[MAX_NUM_LAYER];
				float* temp_weights_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));

				int pro_recv = numprocs;
				if (batch_i * batch_size + t_batch_size < batch_i * batch_size + my_size * numprocs) pro_recv--;

				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数
					temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
					//layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
					//cout << myid << "cal delta@" << j << endl;
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);

					//cout << myid << "cal delta!" << j << endl;
					if (myid != 0)
						MPI_Send(temp_weights_delta, num_each_layer[num_layers], MPI_FLOAT, 0, 100 + j, MPI_COMM_WORLD);
					else {
						int k = 0;
						__m128 t2 = _mm_set1_ps(temp_bias_delta[j]);
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for private(k)
						for (k = 0; k< num_each_layer[num_layers]-4; k += 4)
						{
							__m128 t1, t3, product;
							t1 = _mm_loadu_ps(layers[num_layers]->weights_delta[j] + k);
							t3 = _mm_loadu_ps(layers[num_layers - 1]->output_nodes + k);
							//product=sr*t2*t3，向量对位相乘
							product = _mm_mul_ps(t3, t2);
							t1 = _mm_add_ps(t1, product);
							_mm_storeu_ps(layers[num_layers]->weights_delta[j] + k, t1);
						}
						for (k -= 4; k < num_each_layer[num_layers]; k++)
							layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];

						//for (int k = 0; k < num_each_layer[num_layers]; k++)
						//	layers[num_layers]->weights_delta[j][k] += temp_weights_delta[k];
						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
							MPI_Recv(temp_weights_delta, num_each_layer[num_layers], MPI_FLOAT, MPI_ANY_SOURCE, 100 + j, MPI_COMM_WORLD, &status);
							int k = 0;
							__m128 t2 = _mm_set1_ps(temp_bias_delta[j]);
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for private(k)
							for (k = 0; k< num_each_layer[num_layers]-4; k += 4)
							{
								__m128 t1, t3, product;
								t1 = _mm_loadu_ps(layers[num_layers]->weights_delta[j] + k);
								t3 = _mm_loadu_ps(layers[num_layers - 1]->output_nodes + k);
								//product=sr*t2*t3，向量对位相乘
								product = _mm_mul_ps(t3, t2);
								t1 = _mm_add_ps(t1, product);
								_mm_storeu_ps(layers[num_layers]->weights_delta[j] + k, t1);
							}
							for (k -= 4; k < num_each_layer[num_layers]; k++)
								layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];

							//for (int k = 0; k < num_each_layer[num_layers]; k++)
							//	layers[num_layers]->weights_delta[j][k] += temp_weights_delta[k];
						}
					}
					//MPI_Reduce(temp_weights_delta, layers[num_layers]->weights_delta[j], num_each_layer[num_layers], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
				   // cout << myid << "cal delta#" << j << endl;
				}
				if (myid != 0)
					MPI_Send(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
				else {
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
					for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
						layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
					for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
						MPI_Recv(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
						for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
							layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
					}
				}
				//MPI_Reduce(temp_bias_delta, layers[num_layers]->bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
				//cout << myid << "finish cal delta"<< num_layers << endl;

				//计算每层的delta,行访问优化
				for (int i = num_layers - 1; i >= 0; i--)
				{
					float* error = new float[num_each_layer[i + 1]];

					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
					for (int k = 0; k < num_each_layer[i + 2]; k++)
					{
						int j = 0;
						__m128 t2 = _mm_set1_ps(temp_bias_delta[k]);
						for (; j + 4 < num_each_layer[i + 1]; j += 4)
						{
							__m128 t1, t3, product;
							t1 = _mm_loadu_ps(error + j);
							t3 = _mm_loadu_ps(&layers[i + 1]->weights[k][j]);
							product = _mm_mul_ps(t3, t2);
							t1 = _mm_add_ps(t1, product);
							_mm_storeu_ps(error + j, t1);
						}
						for (j -= 4; j < num_each_layer[i + 1]; j++) {
							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
						}
					}
					for (int j = 0; j < num_each_layer[i + 1]; j++)
					{
						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
						layers[i]->bias_delta[j] += temp_bias_delta[j];
						int k = 0;
						__m128 t2 = _mm_set1_ps(temp_bias_delta[j]);
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for private(k)
						for (k = 0; k< num_each_layer[i]-4; k += 4)
						{
							__m128 t1, t3, product;
							if (i == 0) t3 = _mm_loadu_ps(&_trainMat[index][k]);
							else t3 = _mm_loadu_ps(&layers[i - 1]->output_nodes[k]);
							product = _mm_mul_ps(t3, t2);
							_mm_storeu_ps(&temp_weights_delta[k], product);
						}

						for (k -= 4; k < num_each_layer[i]; k++)
							if (i == 0) temp_weights_delta[k] = temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
							else temp_weights_delta[k] = temp_bias_delta[j] * _trainMat[index][k];

						if (myid != 0)
							MPI_Send(temp_weights_delta, num_each_layer[i], MPI_FLOAT, 0, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD);
						else {
							int k = 0;
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for private(k)
							for (k=0; k< num_each_layer[i]-4; k += 4)
							{
								__m128 t1, t3;
								t3 = _mm_loadu_ps(&temp_weights_delta[k]);
								t1 = _mm_loadu_ps(&layers[i]->weights_delta[j][k]);
								t1 = _mm_add_ps(t3, t1);
								_mm_storeu_ps(&layers[i]->weights_delta[j][k], t1);
							}
							for (k -= 4; k < num_each_layer[i]; k++)
								layers[i]->weights_delta[j][k] += temp_weights_delta[k];

							for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
								//cout << sizeof(temp_weights_delta) << endl;
								MPI_Recv(temp_weights_delta, num_each_layer[i], MPI_FLOAT, MPI_ANY_SOURCE, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD, &status);
								int k = 0;
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for private(k)
								for (k=0; k< num_each_layer[i]-4; k += 4)
								{
									__m128 t1, t3;
									t3 = _mm_loadu_ps(&temp_weights_delta[k]);
									t1 = _mm_loadu_ps(&layers[i]->weights_delta[j][k]);
									t1 = _mm_add_ps(t3, t1);
									_mm_storeu_ps(&layers[i]->weights_delta[j][k], t1);
								}
								for (k -= 4; k < num_each_layer[i]; k++)
									layers[i]->weights_delta[j][k] += temp_weights_delta[k];
							}
						}
						//MPI_Reduce(temp_weights_delta, layers[i]->weights_delta[j], num_each_layer[i], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
					}
					delete[]error;

					if (myid != 0)
						MPI_Send(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, 0, 3 + i, MPI_COMM_WORLD);
					else {
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
						for (int k = 0; k < num_each_layer[i + 1]; k++)
							layers[i]->bias_delta[k] += temp_bias_delta[k];
						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
							MPI_Recv(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_ANY_SOURCE, 3 + i, MPI_COMM_WORLD, &status);
#pragma omp parallel num_threads(NUM_THREADS)
#pragma omp for
							for (int k = 0; k < num_each_layer[i + 1]; k++)
								layers[i]->bias_delta[k] += temp_bias_delta[k];
						}
					}
					//MPI_Reduce(temp_bias_delta, layers[i]->bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
					//cout << myid << "finish cal delta" <<i<< endl;
				}
				std::free(temp_weights_delta);
				std::free(temp_bias_delta);
				//cout << myid << "finish cal delta" << endl;
			}
		
		//cout << myid << "is waiting in bari2" << endl;

		MPI_Barrier(MPI_COMM_WORLD);
		//cout << myid << "after bari2" << endl;
		if (myid == 0) {
			//反向传播，weights和bias更新
			for (int i = 0; i <= num_layers; i++)
			{
#pragma omp for
				for (int k = 0; k < num_each_layer[i + 1]; k++)
				{
					__m128 r = _mm_set1_ps(study_rate / t_batch_size);
					int j = 0;
					for (; j + 4 < num_each_layer[i]; j += 4) {
						__m128 t1, t3, product;
						t1 = _mm_loadu_ps(&layers[i]->weights[k][j]);
						t3 = _mm_loadu_ps(&layers[i]->weights_delta[k][j]);
						product = _mm_mul_ps(t3, r);
						t1 = _mm_sub_ps(t1, product);
						_mm_storeu_ps(&layers[i]->weights[k][j], t1);
					}
					for (j -= 4; j < num_each_layer[i]; j++) {
						layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
					}
					layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
				}
			}
		}
		}
	}
	
	QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
	if (myid == 0) {
		std::cout << myid << "mpi+simd+openmp:" << (tail - head) * 1.0 / freq << "s" << endl;
		std::printf("finish training\n");
	}

}

void ANN_parallel::train_MPI(const int num_sample, float** _trainMat, float** _labelMat)
{
	std::printf("begin training\n");
	long long head, tail, freq;// timers
	long long head1, tail1;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time

	float thre = 1e-2;
	int myid, numprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	for (int epoch = 0; epoch < num_epoch; epoch++)
	{
		//if (myid == 0) isNotConver_(num_sample, _trainMat, _labelMat, thre);
		//fflush(stdout);
		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
			if (myid == 0) {
				for (int i = num_layers; i >= 0; i--) {
					for (int j = 0; j < num_each_layer[i + 1]; j++) {
						memset(layers[i]->weights_delta[j], 0, num_each_layer[i]);
					}
					memset(layers[i]->bias_delta, 0, num_each_layer[i + 1]);
				}
			}
			//cout << myid << "is waiting in bari1" << endl;
			//MPI_Barrier(MPI_COMM_WORLD);
			//cout << myid << "after bari1" << endl;
			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;

			int my_size = (t_batch_size + numprocs - 1) / numprocs;

			for (int index = batch_i * batch_size + my_size * myid;
				index < min(batch_i * batch_size + t_batch_size, batch_i * batch_size + my_size * (myid + 1));
				index++) {
				QueryPerformanceCounter((LARGE_INTEGER*)&head1); // start time
				//cout << myid << "index" <<index<< endl;
				//数据发送
				for (int i_layer = 0; i_layer <= num_layers; i_layer++) {
					MPI_Bcast(layers[i_layer]->bias, num_each_layer[i_layer + 1], MPI_FLOAT, 0, MPI_COMM_WORLD);
					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
						MPI_Bcast(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
				}
				//cout <<myid<< "begin predict" << endl;
				//前向传播
				float output_node;
				for (int i = 0; i < num_each_layer[1]; i++)
				{
					output_node = 0.0;
					for (int j = 0; j < num_each_layer[0]; j++)
					{
						output_node += layers[0]->weights[i][j] * _trainMat[index][j];
					}
					output_node += layers[0]->bias[i];
					layers[0]->output_nodes[i] = layers[0]->activation_function(output_node);//输出-1到1
				}
				for (int i_layer = 1; i_layer <= num_layers; i_layer++)
				{
					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
					{
						output_node = 0.0;
						for (int j = 0; j < num_each_layer[i_layer]; j++)
						{
							output_node += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
						}
						output_node += layers[i_layer]->bias[i];
						layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(output_node);
					}
				}
				//cout <<myid<< "begin cal delta" << endl;
				//计算delta
				MPI_Status status;
				//cout << MAX_NUM_LAYER << endl;
				//for (int i = 0; i < num_layers + 2; i++)cout << num_each_layer[i] << ' ';
				//float* temp_bias_delta = new float[MAX_NUM_LAYER];
				float* temp_bias_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));
				//float* temp_weights_delta = new float[MAX_NUM_LAYER];
				float* temp_weights_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));

				int pro_recv = numprocs;
				if (batch_i * batch_size + t_batch_size < batch_i * batch_size + my_size * numprocs) pro_recv--;

				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
				{
					//均方误差损失函数
					temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
					//layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
					//cout << myid << "cal delta@" << j << endl;
					//交叉熵损失函数
					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
					for (int k = 0; k < num_each_layer[num_layers]; k++)
						temp_weights_delta[k] = temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
					//cout << myid << "cal delta!" << j << endl;
					if (myid != 0)
						MPI_Send(temp_weights_delta, num_each_layer[num_layers], MPI_FLOAT, 0, 100 + j, MPI_COMM_WORLD);
					else {
						for (int k = 0; k < num_each_layer[num_layers]; k++)
							layers[num_layers]->weights_delta[j][k] += temp_weights_delta[k];
						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
							MPI_Recv(temp_weights_delta, num_each_layer[num_layers], MPI_FLOAT, MPI_ANY_SOURCE, 100 + j, MPI_COMM_WORLD, &status);
							for (int k = 0; k < num_each_layer[num_layers]; k++)
								layers[num_layers]->weights_delta[j][k] += temp_weights_delta[k];
						}
					}
					//MPI_Reduce(temp_weights_delta, layers[num_layers]->weights_delta[j], num_each_layer[num_layers], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
				   // cout << myid << "cal delta#" << j << endl;
				}
				if (myid != 0)
					MPI_Send(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
				else {
					for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
						layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
					for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
						MPI_Recv(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
						for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
							layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
					}
				}
				//MPI_Reduce(temp_bias_delta, layers[num_layers]->bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
				//cout << myid << "finish cal delta"<< num_layers << endl;

				//计算每层的delta,行访问优化
				for (int i = num_layers - 1; i >= 0; i--)
				{
					float* error = new float[num_each_layer[i + 1]];

					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
					for (int k = 0; k < num_each_layer[i + 2]; k++)
					{
						for (int j = 0; j < num_each_layer[i + 1]; j++)
						{
							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
						}
					}
					for (int j = 0; j < num_each_layer[i + 1]; j++)
					{
						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
						layers[i]->bias_delta[j] += temp_bias_delta[j];
						for (int k = 0; k < num_each_layer[i]; k++)
							if (i > 0)
								temp_weights_delta[k] = temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
							else
								temp_weights_delta[k] = temp_bias_delta[j] * _trainMat[index][k];
						if (myid != 0)
							MPI_Send(temp_weights_delta, num_each_layer[i], MPI_FLOAT, 0, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD);
						else {
							for (int k = 0; k < num_each_layer[i]; k++)
								layers[i]->weights_delta[j][k] += temp_weights_delta[k];
							for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
								//cout << sizeof(temp_weights_delta) << endl;
								MPI_Recv(temp_weights_delta, num_each_layer[i], MPI_FLOAT, MPI_ANY_SOURCE, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD, &status);
								for (int k = 0; k < num_each_layer[i]; k++)
									layers[i]->weights_delta[j][k] += temp_weights_delta[k];
							}
						}
						//MPI_Reduce(temp_weights_delta, layers[i]->weights_delta[j], num_each_layer[i], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
					}
					delete[]error;

					if (myid != 0)
						MPI_Send(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, 0, 3 + i, MPI_COMM_WORLD);
					else {
						for (int k = 0; k < num_each_layer[i + 1]; k++)
							layers[i]->bias_delta[k] += temp_bias_delta[k];
						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
							MPI_Recv(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_ANY_SOURCE, 3 + i, MPI_COMM_WORLD, &status);
							for (int k = 0; k < num_each_layer[i + 1]; k++)
								layers[i]->bias_delta[k] += temp_bias_delta[k];
						}

					}
				
					//MPI_Reduce(temp_bias_delta, layers[i]->bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
					//cout << myid << "finish cal delta" <<i<< endl;
				}
				std::free(temp_weights_delta);
				std::free(temp_bias_delta);
				//cout << myid << "finish cal delta" << endl;
				QueryPerformanceCounter((LARGE_INTEGER*)&tail1); // start time
				//std::cout << myid << "one index"<<index<<': ' << (tail1 - head1) * 1.0 / freq << "s" << endl;
			}
			//cout << myid << "is waiting in bari2" << endl;

			MPI_Barrier(MPI_COMM_WORLD);
			//cout << myid << "after bari2" << endl;
			if (myid == 0) {
				//反向传播，weights和bias更新
				for (int i = 0; i <= num_layers; i++)
				{
					for (int k = 0; k < num_each_layer[i + 1]; k++)
					{
						for (int j = 0; j < num_each_layer[i]; j++)
						{
							layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
						}
						layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
					}
				}
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail); // end time
	if (myid == 0) {
		std::cout << myid << "mpi_all:" << (tail - head) * 1.0 / freq << "s" << endl;
		std::cout << myid << "one index"  << ': ' << (tail1 - head1) * 1.0 / freq << "s" << endl;
		std::printf("finish training\n");
	}

}

//void ANN_parallel::train_MPI_openMP_SIMD(const int num_sample, float** _trainMat, float** _labelMat)
//{
//	std::printf("begin training\n");
//	long long head, tail, freq;// timers
//	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
//	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start time
//
//	float thre = 1e-2;
//	int myid, numprocs;
//	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
//	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//	for (int epoch = 0; epoch < num_epoch; epoch++)
//	{
//
//		//if (myid == 0) isNotConver_(num_sample, _trainMat, _labelMat, thre);
//		//fflush(stdout);
//		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
//		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
//		#pragma omp parallel num_threads(NUM_THREADS){
//			if (myid == 0) {
//				for (int i = num_layers; i >= 0; i--) {
//#pragma omp parallel for
//					for (int j = 0; j < num_each_layer[i + 1]; j++) {
//						for (int k = 0; k < num_each_layer[i]; k++) {
//							layers[i]->weights_delta[j][k] = 0;
//						}
//						layers[i]->bias_delta[j] = 0;
//					}
//				}
//			}
//			//cout << myid << "is waiting in bari1" << endl;
//			//MPI_Barrier(MPI_COMM_WORLD);
//			//cout << myid << "after bari1" << endl;
//			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
//			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;
//
//			int my_size = (t_batch_size + numprocs - 1) / numprocs;
//
//
//			for (int index = batch_i * batch_size + my_size * myid;
//				index < min(batch_i * batch_size + t_batch_size, batch_i * batch_size + my_size * (myid + 1));
//				index++) {
//				//cout << myid << "index" <<index<< endl;
//				//数据发送
//				for (int i_layer = 0; i_layer <= num_layers; i_layer++) {
//					MPI_Bcast(layers[i_layer]->bias, num_each_layer[i_layer + 1], MPI_FLOAT, 0, MPI_COMM_WORLD);
//					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
//						MPI_Bcast(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
//				}
//				//cout <<myid<< "begin predict" << endl;
//				//前向传播
//
//#pragma omp parallel for
//				for (int i = 0; i < num_each_layer[1]; i++)
//				{
//					float output_node = 0.0;
//					for (int j = 0; j < num_each_layer[0]; j++)
//					{
//						output_node += layers[0]->weights[i][j] * _trainMat[index][j];
//					}
//					output_node += layers[0]->bias[i];
//					layers[0]->output_nodes[i] = layers[0]->activation_function(output_node);//输出-1到1
//				}
//				for (int i_layer = 1; i_layer <= num_layers; i_layer++)
//				{
//#pragma omp parallel for
//					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
//					{
//						float output_node = 0.0;
//						for (int j = 0; j < num_each_layer[i_layer]; j++)
//						{
//							output_node += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
//						}
//						output_node += layers[i_layer]->bias[i];
//						layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(output_node);
//					}
//				}
//				//cout <<myid<< "begin cal delta" << endl;
//				//计算delta
//				MPI_Status status;
//				//cout << MAX_NUM_LAYER << endl;
//				//for (int i = 0; i < num_layers + 2; i++)cout << num_each_layer[i] << ' ';
//				//float* temp_bias_delta = new float[MAX_NUM_LAYER];
//				float* temp_bias_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));
//				//float* temp_weights_delta = new float[MAX_NUM_LAYER];
//				float* temp_weights_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));
//
//				int pro_recv = numprocs;
//				if (batch_i * batch_size + t_batch_size < batch_i * batch_size + my_size * numprocs) pro_recv--;
//
//				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
//				{
//					//均方误差损失函数
//					temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
//					//layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
//					//cout << myid << "cal delta@" << j << endl;
//					//交叉熵损失函数
//					//layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
//#pragma omp parallel for
//					for (int k = 0; k < num_each_layer[num_layers]; k++)
//						temp_weights_delta[k] = temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
//					//cout << myid << "cal delta!" << j << endl;
//					if (myid != 0)
//						MPI_Send(temp_weights_delta, num_each_layer[num_layers], MPI_FLOAT, 0, 100 + j, MPI_COMM_WORLD);
//					else {
//#pragma omp parallel for
//						for (int k = 0; k < num_each_layer[num_layers]; k++)
//							layers[num_layers]->weights_delta[j][k] += temp_weights_delta[k];
//						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//							MPI_Recv(temp_weights_delta, num_each_layer[num_layers], MPI_FLOAT, MPI_ANY_SOURCE, 100 + j, MPI_COMM_WORLD, &status);
//						}
//#pragma omp parallel for
//						for (int k = 0; k < num_each_layer[num_layers]; k++)
//							layers[num_layers]->weights_delta[j][k] += temp_weights_delta[k];
//					}
//					//MPI_Reduce(temp_weights_delta, layers[num_layers]->weights_delta[j], num_each_layer[num_layers], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//				   // cout << myid << "cal delta#" << j << endl;
//				}
//				if (myid != 0)
//					MPI_Send(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
//				else {
//#pragma omp parallel for
//					for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
//						layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
//					for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//						MPI_Recv(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
//					}
//#pragma omp parallel for
//					for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
//						layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
//				}
//				//MPI_Reduce(temp_bias_delta, layers[num_layers]->bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//				//cout << myid << "finish cal delta"<< num_layers << endl;
//
//				//计算每层的delta,行访问优化
//				for (int i = num_layers - 1; i >= 0; i--)
//				{
//					float* error = new float[num_each_layer[i + 1]];
//
//					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
//					for (int k = 0; k < num_each_layer[i + 2]; k++)
//					{
//#pragma omp parallel for
//						for (int j = 0; j < num_each_layer[i + 1]; j++)
//						{
//							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
//						}
//					}
//					for (int j = 0; j < num_each_layer[i + 1]; j++)
//					{
//						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
//						layers[i]->bias_delta[j] += temp_bias_delta[j];
//#pragma omp parallel for
//						for (int k = 0; k < num_each_layer[i]; k++)
//							if (i > 0)
//								temp_weights_delta[k] = temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
//							else
//								temp_weights_delta[k] = temp_bias_delta[j] * _trainMat[index][k];
//						if (myid != 0)
//							MPI_Send(temp_weights_delta, num_each_layer[i], MPI_FLOAT, 0, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD);
//						else {
//#pragma omp parallel for
//							for (int k = 0; k < num_each_layer[i]; k++)
//								layers[i]->weights_delta[j][k] += temp_weights_delta[k];
//							for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//								//cout << sizeof(temp_weights_delta) << endl;
//								MPI_Recv(temp_weights_delta, num_each_layer[i], MPI_FLOAT, MPI_ANY_SOURCE, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD, &status);
//							}
//#pragma omp parallel for
//							for (int k = 0; k < num_each_layer[i]; k++)
//								layers[i]->weights_delta[j][k] += temp_weights_delta[k];
//						}
//						//MPI_Reduce(temp_weights_delta, layers[i]->weights_delta[j], num_each_layer[i], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//					}
//					delete[]error;
//
//					if (myid != 0)
//						MPI_Send(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, 0, 3 + i, MPI_COMM_WORLD);
//					else {
//						for (int k = 0; k < num_each_layer[i + 1]; k++)
//							layers[i]->bias_delta[k] += temp_bias_delta[k];
//						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//							MPI_Recv(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_ANY_SOURCE, 3 + i, MPI_COMM_WORLD, &status);
//						}
//						for (int k = 0; k < num_each_layer[i + 1]; k++)
//							layers[i]->bias_delta[k] += temp_bias_delta[k];
//					}
//					//MPI_Reduce(temp_bias_delta, layers[i]->bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//					//cout << myid << "finish cal delta" <<i<< endl;
//				}
//				std::free(temp_weights_delta);
//				std::free(temp_bias_delta);
//				//cout << myid << "finish cal delta" << endl;
//			}
//			//cout << myid << "is waiting in bari2" << endl;
//
//			MPI_Barrier(MPI_COMM_WORLD);
//			//cout << myid << "after bari2" << endl;
//			if (myid == 0) {
//				//反向传播，weights和bias更新
//				for (int i = 0; i <= num_layers; i++)
//				{
//#pragma omp parallel for
//					for (int k = 0; k < num_each_layer[i + 1]; k++)
//					{
//						for (int j = 0; j < num_each_layer[i]; j++)
//						{
//							layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
//						}
//						layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
//					}
//				}
//			}
//		}
//		}
//	}
//	
//	//QueryPerformanceCounter((LARGE_INTEGER*)&tail);// end time
//
//	//if (myid == 0) {
//	//	std::cout << myid << "mpi_all:" << (tail - head) * 1.0 / freq << "s" << endl;
//	//	std::printf("finish training\n");
//	//}
//}

//
//void ANN_parallel::train_MPI_i(const int num_sample, float** _trainMat, float** _labelMat)
//{
//	std::printf("begin training\n");
//	float thre = 1e-2;
//	int myid, numprocs;
//	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
//	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//	for (int epoch = 0; epoch < num_epoch; epoch++)
//	{
//		if (myid == 0)
//			isNotConver_(num_sample, _trainMat, _labelMat, thre);
//		fflush(stdout);
//		// if (epoch % 50 == 0) printf ("round%d:\n", epoch);
//		for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
//			if (myid == 0) {
//				for (int i = num_layers; i >= 0; i--)
//					for (int j = 0; j < num_each_layer[i + 1]; j++) {
//						for (int k = 0; k < num_each_layer[i]; k++) {
//							layers[i]->weights_delta[j][k] = 0;
//						}
//						layers[i]->bias_delta[j] = 0;
//					}
//			}
//			//cout << myid << "is waiting in bari1" << endl;
//			//MPI_Barrier(MPI_COMM_WORLD);
//			//cout << myid << "after bari1" << endl;
//			int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
//			if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;
//
//			int my_size = (t_batch_size + numprocs - 1) / numprocs;
//
//			for (int index = batch_i * batch_size + my_size * myid;
//				index < min(batch_i * batch_size + t_batch_size, batch_i * batch_size + my_size * (myid + 1));
//				index++) {
//				//cout << myid << "index" <<index<< endl;
//				//数据发送
//				for (int i_layer = 0; i_layer <= num_layers; i_layer++) {
//					MPI_Bcast(layers[i_layer]->bias, num_each_layer[i_layer + 1], MPI_FLOAT, 0, MPI_COMM_WORLD);
//					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
//						MPI_Bcast(layers[i_layer]->weights[i], num_each_layer[i_layer], MPI_FLOAT, 0, MPI_COMM_WORLD);
//				}
//				//cout <<myid<< "begin predict" << endl;
//				//前向传播
//				float output_node;
//				for (int i = 0; i < num_each_layer[1]; i++)
//				{
//					output_node = 0.0;
//					for (int j = 0; j < num_each_layer[0]; j++)
//					{
//						output_node += layers[0]->weights[i][j] * _trainMat[index][j];
//					}
//					output_node += layers[0]->bias[i];
//					layers[0]->output_nodes[i] = layers[0]->activation_function(output_node);//输出-1到1
//				}
//				for (int i_layer = 1; i_layer <= num_layers; i_layer++)
//				{
//					for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
//					{
//						output_node = 0.0;
//						for (int j = 0; j < num_each_layer[i_layer]; j++)
//						{
//							output_node += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
//						}
//						output_node += layers[i_layer]->bias[i];
//						layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(output_node);
//					}
//				}
//				//cout <<myid<< "begin cal delta" << endl;
//				//计算delta
//				MPI_Status status;
//				MPI_Request request;
//				//cout << MAX_NUM_LAYER << endl;
//				//for (int i = 0; i < num_layers + 2; i++)cout << num_each_layer[i] << ' ';
//				//float* temp_bias_delta = new float[MAX_NUM_LAYER];
//				float* temp_bias_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));
//				//float* temp_weights_delta = new float[MAX_NUM_LAYER];
//				float* temp_weights_delta = (float*)malloc(MAX_NUM_LAYER * sizeof(float));
//
//				int pro_recv = numprocs;
//				if (batch_i * batch_size + t_batch_size < batch_i * batch_size + my_size * numprocs) pro_recv--;
//
//				for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
//				{
//					if (myid != 0) {
//						temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]);
//						for (int k = 0; k < num_each_layer[num_layers]; k++)
//							layers[num_layers]->weights_delta[j][k] = temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
//						MPI_Isend(layers[num_layers]->weights_delta[j], num_each_layer[num_layers], MPI_FLOAT, 0, 100 + j, MPI_COMM_WORLD, &request);
//					}
//					else {
//						temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]);
//						for (int k = 0; k < num_each_layer[num_layers]; k++)
//							layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
//						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//							MPI_Irecv(temp_weights_delta, num_each_layer[num_layers], MPI_FLOAT, MPI_ANY_SOURCE, 100 + j, MPI_COMM_WORLD, &request);
//						}
//						for (int k = 0; k < num_each_layer[num_layers]; k++)
//							layers[num_layers]->weights_delta[j][k] += temp_weights_delta[k];
//					}
//					//MPI_Reduce(temp_weights_delta, layers[num_layers]->weights_delta[j], num_each_layer[num_layers], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//				   // cout << myid << "cal delta#" << j << endl;
//				}
//				if (myid != 0)
//					MPI_Send(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
//				else {
//					for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
//						layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
//					for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//						MPI_Recv(temp_bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
//					}
//					for (int k = 0; k < num_each_layer[num_layers + 1]; k++)
//						layers[num_layers]->bias_delta[k] += temp_bias_delta[k];
//				}
//				//MPI_Reduce(temp_bias_delta, layers[num_layers]->bias_delta, num_each_layer[num_layers + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//				//cout << myid << "finish cal delta"<< num_layers << endl;
//
//				//计算每层的delta,行访问优化
//				for (int i = num_layers - 1; i >= 0; i--)
//				{
//					float* error = new float[num_each_layer[i + 1]];
//
//					for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
//					for (int k = 0; k < num_each_layer[i + 2]; k++)
//					{
//						for (int j = 0; j < num_each_layer[i + 1]; j++)
//						{
//							error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
//						}
//					}
//					for (int j = 0; j < num_each_layer[i + 1]; j++)
//					{
//						temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
//						layers[i]->bias_delta[j] += temp_bias_delta[j];
//						for (int k = 0; k < num_each_layer[i]; k++)
//							if (i > 0)
//								temp_weights_delta[k] = temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
//							else
//								temp_weights_delta[k] = temp_bias_delta[j] * _trainMat[index][k];
//						if (myid != 0)
//							MPI_Send(temp_weights_delta, num_each_layer[i], MPI_FLOAT, 0, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD);
//						else {
//							for (int k = 0; k < num_each_layer[i]; k++)
//								layers[i]->weights_delta[j][k] += temp_weights_delta[k];
//							for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//								//cout << sizeof(temp_weights_delta) << endl;
//								MPI_Recv(temp_weights_delta, num_each_layer[i], MPI_FLOAT, MPI_ANY_SOURCE, 100 + MAX_NUM_LAYER + i * MAX_NUM_LAYER + j, MPI_COMM_WORLD, &status);
//							}
//							for (int k = 0; k < num_each_layer[i]; k++)
//								layers[i]->weights_delta[j][k] += temp_weights_delta[k];
//						}
//						//MPI_Reduce(temp_weights_delta, layers[i]->weights_delta[j], num_each_layer[i], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//					}
//					delete[]error;
//					if (myid != 0)
//						MPI_Send(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, 0, 3 + i, MPI_COMM_WORLD);
//					else {
//						for (int k = 0; k < num_each_layer[i + 1]; k++)
//							layers[i]->bias_delta[k] += temp_bias_delta[k];
//						for (int pro_i = 1; pro_i < pro_recv; pro_i++) {
//							MPI_Recv(temp_bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_ANY_SOURCE, 3 + i, MPI_COMM_WORLD, &status);
//						}
//						for (int k = 0; k < num_each_layer[i + 1]; k++)
//							layers[i]->bias_delta[k] += temp_bias_delta[k];
//					}
//					//MPI_Reduce(temp_bias_delta, layers[i]->bias_delta, num_each_layer[i + 1], MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
//					//cout << myid << "finish cal delta" <<i<< endl;
//				}
//				free(temp_weights_delta);
//				free(temp_bias_delta);
//				//cout << myid << "finish cal delta" << endl;
//			}
//			//cout << myid << "is waiting in bari2" << endl;
//
//			MPI_Barrier(MPI_COMM_WORLD);
//			//cout << myid << "after bari2" << endl;
//			if (myid == 0) {
//				//反向传播，weights和bias更新
//				for (int i = 0; i <= num_layers; i++)
//				{
//					for (int k = 0; k < num_each_layer[i + 1]; k++)
//					{
//						for (int j = 0; j < num_each_layer[i]; j++)
//						{
//							layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
//						}
//						layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
//					}
//				}
//			}
//		}
//	}
//	std::printf("finish training\n");
//}

bool ANN_parallel::isNotConver_(const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh)
{
	float lossFunc = 0.0;
	for (int k = 0; k < _sampleNum; ++k)
	{
		predict(_trainMat[k]);
		float loss = 0.0;
		for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
		{
			loss += (layers[num_layers]->output_nodes[t] - _labelMat[k][t]) * (layers[num_layers]->output_nodes[t] - _labelMat[k][t]);
		}
		lossFunc += (1.0 / 2) * loss;
	}

	lossFunc = lossFunc / _sampleNum;

	static int tt = 0;
	tt++;
	printf("第%d次训练：%0.12f\n", tt, lossFunc);

	if (lossFunc > _thresh) return true;
	return false;
}

void ANN_parallel::predict(float* in)
{
	layers[0]->_forward(in);
	for (int i = 1; i <= num_layers; i++)
	{
		layers[i]->_forward(layers[i - 1]->output_nodes);
	}
}
void ANN_parallel::display()
{
	for (int i = 0; i <= num_layers; i++)
	{
		layers[i]->display();
	}
}

void ANN_parallel::show_predictions(float* X)
{
	predict(X);
	static int t = 0;
	printf("in%d:", t);
	for (int i = 0; i < min(5, num_each_layer[0]); i++) printf("%f,", X[i]);
	printf("  out%d:", t);
	for (int i = 0; i < min(5, num_each_layer[num_layers + 1]); i++)
		printf("%f,", layers[num_layers]->output_nodes[i]);
	t++;
	printf("\n");
	//return layers[num_layers]->output_nodes;
}
