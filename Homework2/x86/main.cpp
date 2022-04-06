#include <iostream>
#include<ANN.h>
#include<ANN_SIMD.h>
#include<windows.h>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
	const int hidnodes = 1024; //单层隐藏层的结点数
	const int inNodes = 128;   //输入层结点数
	const int outNodes = 128;  //输出层结点数

	const int trainClass = 8; //类别数
	const int numPerClass = 32;  //每个类别的样本点数

	int sampleN = trainClass * numPerClass;     //总的样本数=每类训练样本数*类别数
	float** trainMat = new float* [sampleN];                         //生成训练样本
	for (int k = 0; k < trainClass; ++k) {
		for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i) {
			trainMat[i] = new float[inNodes];
			for (int j = 0; j < inNodes; ++j) {
				trainMat[i][j] = rand() % 1000 / 10000.0 + 0.1 * (2 * k + 1);

			}
		}
	}

	float** labelMat = new float* [sampleN]; //生成标签矩阵
	for (int k = 0; k < trainClass; ++k) {
		for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i) {
			labelMat[i] = new float[outNodes];
			for (int j = 0; j < trainClass; ++j) {
				if (j == k)
					labelMat[i][j] = 1;
				else
					labelMat[i][j] = 0;
			}

		}
	}
    long long head, tail, freq;// timers
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

   // /*
    ANN_SIMD ann_simd_classify3(sampleN, inNodes, outNodes, hidnodes, 0.12);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
	ann_simd_classify3.train_avx(sampleN, trainMat, labelMat);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
    cout <<"avx:"<< (tail - head) * 1000.0 / freq << "ms" << endl;
    //*/
	ANN_SIMD ann_simd_classify1(sampleN, inNodes, outNodes, hidnodes, 0.12);  //输入层为inNodes个结点，输出层outNodes个结点，单层隐藏层,studyRate为0.12

    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
	ann_simd_classify1.train_sse(sampleN, trainMat, labelMat);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
    cout <<"sse:"<< (tail - head) * 1000.0 / freq << "ms" << endl;

    ANN_SIMD ann_simd_classify2(sampleN, inNodes, outNodes, hidnodes, 0.12);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
	ann_simd_classify2.train_cache(sampleN, trainMat, labelMat);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
    cout <<"cache:"<< (tail - head) * 1000.0 / freq << "ms" << endl;



    ANN ann_classify(sampleN, inNodes, outNodes, hidnodes, 0.12);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
	ann_classify.train(sampleN, trainMat, labelMat);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
    cout <<"original:"<< (tail - head) * 1000.0 / freq << "ms" << endl;

	for (int i = 0; i < 30; ++i) {
		ann_classify.predict(trainMat[i + 120], NULL);
		//std::cout << std::endl;
	}


	//释放内存
	for (int i = 0; i < sampleN; ++i)
		delete[] trainMat[i];
	delete[] trainMat;

	for (int i = 0; i < sampleN; ++i)
		delete[] labelMat[i];
	delete[] labelMat;

	return 0;
}
