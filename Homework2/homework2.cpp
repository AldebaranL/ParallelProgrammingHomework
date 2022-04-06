#include <iostream>
#include "ANN.cpp"
#include "ANN_SIMD.cpp"
#include <unistd.h>
#include <sys/time.h>
#include <cstdlib>
#include <ctime>
using namespace std;

int main()
{
	const int hidnodes = 128; //单层隐藏层的结点数
	const int inNodes = 128;   //输入层结点数
	const int outNodes = 128;  //输出层结点数

	const int trainClass = 8; //5个类别
	const int numPerClass = 32;  //每个类别30个样本点

	int sampleN = trainClass * numPerClass;     //每类训练样本数为30，5个类别，总的样本数为150

//===========================init begin======================================================
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
//===========================init end======================================================
	struct timespec sts,ets;
	time_t dsec;
	long dnsec;

	ANN_SIMD ann_simd_classify(sampleN, inNodes, outNodes, hidnodes, 0.12);
	timespec_get(&sts, TIME_UTC);
	ann_simd_classify.train_neon(sampleN, trainMat, labelMat);
    timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("%ld.%09lds\n",dsec,dnsec);

	ANN ann_classify(sampleN, inNodes, outNodes, hidnodes, 0.12); 
	timespec_get(&sts, TIME_UTC);
	ann_classify.train(sampleN, trainMat, labelMat);
	timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("%ld.%09lds\n",dsec,dnsec);

	for (int i = 0; i < 30; ++i) {
		ann_classify.predict(trainMat[i + 120], NULL);
		//std::cout << std::endl;
	}

//===========================delete begin======================================================
	//释放内存
	for (int i = 0; i < sampleN; ++i)
		delete[] trainMat[i];
	delete[] trainMat;

	for (int i = 0; i < sampleN; ++i)
		delete[] labelMat[i];
	delete[] labelMat;
//===========================delete end======================================================
	return 0;
}
