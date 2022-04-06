#include "ANN.h"


ANN::ANN(int _SampleN, int nNIL, int nNOL, const int nNHL, float _sR) :
	SampleCount(_SampleN), numNodesInputLayer(nNIL), numNodesOutputLayer(nNOL),
	numNodesHiddenLayer(nNHL), studyRate(_sR)
{
    MAXTT=16;
	//创建权值空间,并初始化
	srand(time(NULL));
	weights = new float** [2];
	weights[0] = new float* [numNodesInputLayer];
	for (int i = 0; i < numNodesInputLayer; ++i) {
		weights[0][i] = new float[numNodesHiddenLayer];
		for (int j = 0; j < numNodesHiddenLayer; ++j) {
			weights[0][i][j] = (rand() % (2000) / 1000.0 - 1.0); //-1到1之间
		}
	}
	weights[1] = new float* [numNodesHiddenLayer];
	for (int i = 0; i < numNodesHiddenLayer; ++i) {
		weights[1][i] = new float[numNodesOutputLayer];
		for (int j = 0; j < numNodesOutputLayer; ++j) {
			weights[1][i][j] = (rand() % (2000) / 1000.0 - 1.0); //-1到1之间
		}
	}

	//创建偏置空间，并初始化
	bias = new float* [2];
	bias[0] = new float[numNodesHiddenLayer];
	for (int i = 0; i < numNodesHiddenLayer; ++i) {
		bias[0][i] = (rand() % (2000) / 1000.0 - 1.0); //-1到1之间
	}
	bias[1] = new float[numNodesOutputLayer];
	for (int i = 0; i < numNodesOutputLayer; ++i) {
		bias[1][i] = (rand() % (2000) / 1000.0 - 1.0); //-1到1之间
	}

	//创建隐藏层各结点的输出值空间
	hidenLayerOutput = new float[numNodesHiddenLayer];
	//创建输出层各结点的输出值空间
	outputLayerOutput = new float[numNodesOutputLayer];

	//创建所有样本的权值更新量存储空间
	allDeltaWeights = new float*** [_SampleN];
	for (int k = 0; k < _SampleN; ++k) {
		allDeltaWeights[k] = new float** [2];
		allDeltaWeights[k][0] = new float* [numNodesInputLayer];
		for (int i = 0; i < numNodesInputLayer; ++i) {
			allDeltaWeights[k][0][i] = new float[numNodesHiddenLayer];
		}
		allDeltaWeights[k][1] = new float* [numNodesHiddenLayer];
		for (int i = 0; i < numNodesHiddenLayer; ++i) {
			allDeltaWeights[k][1][i] = new float[numNodesOutputLayer];
		}
	}

	//创建所有样本的偏置更新量存储空间
	allDeltaBias = new float** [_SampleN];
	for (int k = 0; k < _SampleN; ++k) {
		allDeltaBias[k] = new float* [2];
		allDeltaBias[k][0] = new float[numNodesHiddenLayer];
		allDeltaBias[k][1] = new float[numNodesOutputLayer];
	}

	//创建存储所有样本的输出层输出空间
	outputMat = new float* [_SampleN];
	for (int k = 0; k < _SampleN; ++k) {
		outputMat[k] = new float[numNodesOutputLayer];
	}


}

ANN::~ANN()
{
	//释放权值空间
	for (int i = 0; i < numNodesInputLayer; ++i)
		delete[] weights[0][i];
	for (int i = 1; i < numNodesHiddenLayer; ++i)
		delete[] weights[1][i];
	for (int i = 0; i < 2; ++i)
		delete[] weights[i];
	delete[] weights;

	//释放偏置空间
	for (int i = 0; i < 2; ++i)
		delete[] bias[i];
	delete[] bias;

	//释放所有样本的权值更新量存储空间
	for (int k = 0; k < SampleCount; ++k) {
		for (int i = 0; i < numNodesInputLayer; ++i)
			delete[] allDeltaWeights[k][0][i];
		for (int i = 1; i < numNodesHiddenLayer; ++i)
			delete[] allDeltaWeights[k][1][i];
		for (int i = 0; i < 2; ++i)
			delete[] allDeltaWeights[k][i];
		delete[] allDeltaWeights[k];
	}
	delete[] allDeltaWeights;

	//释放所有样本的偏置更新量存储空间
	for (int k = 0; k < SampleCount; ++k) {
		for (int i = 0; i < 2; ++i)
			delete[] allDeltaBias[k][i];
		delete[] allDeltaBias[k];
	}
	delete[] allDeltaBias;

	//释放存储所有样本的输出层输出空间
	for (int k = 0; k < SampleCount; ++k)
		delete[] outputMat[k];
	delete[] outputMat;

}

void ANN::train(const int _sampleNum, float** _trainMat, float** _labelMat)
{
	float thre = 1e-2;
	for (int i = 0; i < _sampleNum; ++i) {
		train_vec(_trainMat[i], _labelMat[i], i);
	}
	for(int tt = 0; tt < MAXTT; tt++){
		if(!isNotConver_(_sampleNum, _labelMat, thre)) break;
		//调整权值
		for (int index = 0; index < _sampleNum; ++index) {
			for (int i = 0; i < numNodesInputLayer; ++i) {
				for (int j = 0; j < numNodesHiddenLayer; ++j) {
					weights[0][i][j] -= studyRate * allDeltaWeights[index][0][i][j];
				}
			}
			for (int i = 0; i < numNodesHiddenLayer; ++i) {
				for (int j = 0; j < numNodesOutputLayer; ++j) {
					weights[1][i][j] -= studyRate * allDeltaWeights[index][1][i][j];
				}
			}
		}

		for (int index = 0; index < _sampleNum; ++index) {
			for (int i = 0; i < numNodesHiddenLayer; ++i) {
				bias[0][i] -= studyRate * allDeltaBias[index][0][i];
			}
			for (int i = 0; i < numNodesOutputLayer; ++i) {
				bias[1][i] -= studyRate * allDeltaBias[index][1][i];
			}
		}

		for (int i = 0; i < _sampleNum; ++i) {
			train_vec(_trainMat[i], _labelMat[i], i);
		}
	}

	//printf("训练权值和偏置成功了！\n");
}

void ANN::train_vec(const float* _trainVec, const float* _labelVec, int index)
{
	//计算各隐藏层结点的输出
	for (int i = 0; i < numNodesHiddenLayer; ++i) {
		float z = 0.0;
		for (int j = 0; j < numNodesInputLayer; ++j) {
			z += _trainVec[j] * weights[0][j][i];
		}
		z += bias[0][i];
		hidenLayerOutput[i] = sigmoid(z);

	}

	//计算输出层结点的输出值
	for (int i = 0; i < numNodesOutputLayer; ++i) {
		float z = 0.0;
		for (int j = 0; j < numNodesHiddenLayer; ++j) {
			z += hidenLayerOutput[j] * weights[1][j][i];
		}
		z += bias[1][i];
		outputLayerOutput[i] = sigmoid(z);
		outputMat[index][i] = outputLayerOutput[i];
	}

	//计算偏置及权重更新量，但不更新

	for (int j = 0; j < numNodesOutputLayer; ++j) {
		allDeltaBias[index][1][j] = (-0.1) * (_labelVec[j] - outputLayerOutput[j]) * outputLayerOutput[j]
			* (1 - outputLayerOutput[j]);
		for (int i = 0; i < numNodesHiddenLayer; ++i) {
			allDeltaWeights[index][1][i][j] = allDeltaBias[index][1][j] * hidenLayerOutput[i];
		}
	}
	for (int j = 0; j < numNodesHiddenLayer; ++j) {
		float z = 0.0;
		for (int k = 0; k < numNodesOutputLayer; ++k) {
			z += weights[1][j][k] * allDeltaBias[index][1][k];
		}
		allDeltaBias[index][0][j] = z * hidenLayerOutput[j] * (1 - hidenLayerOutput[j]);
		for (int i = 0; i < numNodesInputLayer; ++i) {
			allDeltaWeights[index][0][i][j] = allDeltaBias[index][0][j] * _trainVec[i];
		}
	}

}


bool ANN::isNotConver_(const int _sampleNum,
	float** _labelMat, float _thresh)
{
	float lossFunc = 0.0;
	for (int k = 0; k < _sampleNum; ++k) {
		float loss = 0.0;
		for (int t = 0; t < numNodesOutputLayer; ++t) {
			loss += (outputMat[k][t] - _labelMat[k][t]) * (outputMat[k][t] - _labelMat[k][t]);
		}
		lossFunc += (1.0 / 2) * loss;
	}

	lossFunc = lossFunc / _sampleNum;

	//for (int k = 0; k < _sampleNum; ++k){
	//	for (int i = 0; i< numNodesOutputLayer; ++i){
	//		std::cout << outputMat[k][i] << " " ;
	//	}
	//	std::cout << std::endl;
	//}

	///*
	//第几次时的损失函数值//
	static int tt = 0;
	tt++;
	//if (tt % 1000 == 0) {
		printf("第%d次训练：", tt);
		printf("%0.12f\n", lossFunc);
	//}
	//*/


	if (lossFunc > _thresh)
		return true;

	return false;
}

void ANN::predict(float* in, float* proba)
{
	//输出训练得到的权值
		//std::cout << "\n输出训练得到的权值:\n";
		//for (int i = 0; i < numNodesInputLayer; ++i){
		//	for (int j = 0; j < numNodesHiddenLayer; ++j)
		//		std::cout <<weights[0][i][j] << " ";
		//}
		//std::cout << "\n\n\n";
		//for (int i = 0; i < numNodesHiddenLayer; ++i){
		//	for (int j = 0; j < numNodesOutputLayer; ++j)
		//		std::cout<< weights[1][i][j] << " ";
		//}
		//std::cout << "\n输出训练得到的偏置:\n";
		//for (int i = 0; i < numNodesHiddenLayer; ++i)
		//	std::cout << bias[0][i] << " ";
		//std::cout << "\n\n\n";
		//for (int j = 0; j < numNodesOutputLayer; ++j)
		//	std::cout << bias[1][j] << " ";
		//Sleep(5000);

		//计算各隐藏层结点的输出
	for (int i = 0; i < numNodesHiddenLayer; ++i) {
		float z = 0.0;
		for (int j = 0; j < numNodesInputLayer; ++j) {
			z += in[j] * weights[0][j][i];
		}
		z += bias[0][i];
		hidenLayerOutput[i] = sigmoid(z);
	}

	//计算输出层结点的输出值
	for (int i = 0; i < numNodesOutputLayer; ++i) {
		float z = 0.0;
		for (int j = 0; j < numNodesHiddenLayer; ++j) {
			z += hidenLayerOutput[j] * weights[1][j][i];
		}
		z += bias[1][i];
		outputLayerOutput[i] = sigmoid(z);
		//if ((i % 100) == 0)
		//	std::cout << outputLayerOutput[i] << " ";
	}

}
