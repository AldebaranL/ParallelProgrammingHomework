#include "ANN_SIMD.h"


ANN_SIMD::ANN_SIMD(int _SampleN, int nNIL, int nNOL, const int nNHL, float _sR) :
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

ANN_SIMD::~ANN_SIMD()
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

void ANN_SIMD::train_neon(const int _sampleNum, float** _trainMat, float** _labelMat)
{
	float thre = 1e-2;
	for (int i = 0; i < _sampleNum; ++i) {
		train_vec_neon(_trainMat[i], _labelMat[i], i);
	}
	for(int tt = 0; tt < MAXTT; tt++){
		if(!isNotConver_(_sampleNum, _labelMat, thre)) break;
		//调整权值
		float32x4_t sr = vmovq_n_f32(studyRate);
        float32x4_t t1, t2;
		for (int index = 0; index < _sampleNum; ++index) {
			for (int i = 0; i < numNodesInputLayer; ++i) {
				for (int j = 0; j < numNodesHiddenLayer-4; j+=4) {
					t1 = vld1q_f32(allDeltaWeights[index][0][i] + j);
				    t1 = vmulq_f32(t1, sr);
				    t2 = vld1q_f32(weights[0][i] + j);
				    t2 = vsubq_f32(t2,t1);
                    vst1q_f32(weights[0][i] + j, t2);
					//weights[0][i][j] -= studyRate * allDeltaWeights[index][0][i][j];
				}
			}
			for (int i = 0; i < numNodesHiddenLayer; ++i) {
				for (int j = 0; j < numNodesOutputLayer-4; j+=4) {
                    t1 = vld1q_f32(allDeltaWeights[index][1][i] + j);
				    t1 = vmulq_f32(t1, sr);
				    t2 = vld1q_f32(weights[1][i] + j);
				    t2 = vsubq_f32(t2,t1);
                    vst1q_f32(weights[1][i] + j, t2);
					//weights[1][i][j] -= studyRate * allDeltaWeights[index][1][i][j];
				}
			}
		}
		for (int index = 0; index < _sampleNum; ++index) {
			for (int i = 0; i < numNodesHiddenLayer-4; i+=4) {
                t1 = vld1q_f32(allDeltaBias[index][0]+i);
                t1 = vmulq_f32(t1, sr);
                t2 = vld1q_f32(bias[0] + i);
                t2 = vsubq_f32(t2,t1);
                vst1q_f32(bias[0] + i, t2);
				//bias[0][i] -= studyRate * allDeltaBias[index][0][i];
			}
			for (int i = 0; i < numNodesOutputLayer-4; i+=4) {
			    t1 = vld1q_f32(allDeltaBias[index][1]+i);
                t1 = vmulq_f32(t1, sr);
                t2 = vld1q_f32(bias[1] + i);
                t2 = vsubq_f32(t2,t1);
                vst1q_f32(bias[1] + i, t2);
				//bias[1][i] -= studyRate * allDeltaBias[index][1][i];
			}
		}

		for (int i = 0; i < _sampleNum; ++i) {
			train_vec_neon(_trainMat[i], _labelMat[i], i);
		}
    }
}

void ANN_SIMD::train_vec_neon(const float* _trainVec, const float* _labelVec, int index)
{

	//计算各隐藏层结点的输出
    for (int i = 0; i < numNodesHiddenLayer; ++i) {
        float z = 0.0;
        float32x4_t sums = vmovq_n_f32(0);
        // 内层循环使用SSE(AVX)每次处理4(8)个float数据
        for (int j = 0; j < numNodesInputLayer - 4; j += 4) {
            float32x4_t vecs = vld1q_f32(_trainVec + j);
            // weights不连续，需各个单独设置，注意高低位
			float32x4_t weights128;
			vld1q_lane_f32(weights[0][j]+i, weights128, 0);
			vld1q_lane_f32(weights[0][j+1]+i, weights128, 1);
			vld1q_lane_f32(weights[0][j+2]+i, weights128, 2);
			vld1q_lane_f32(weights[0][j+3]+i, weights128, 3);
			//vset_lane_f32(weights[0][j + 1][i], weights128, 1);
			//vset_lane_f32(weights[0][j + 2][i], weights128, 2);
			//vset_lane_f32(weights[0][j + 3][i], weights128, 3);
            // 乘加指令
			weights128 = vaddq_f32(vecs,weights128);
			sums = vmulq_f32(weights128, sums);
        }
        // 4个局部和相加
		float32x2_t suml2=vget_low_f32(sums);
		// 将高位两个元素保存到sumh2向量
		float32x2_t sumh2=vget_high_f32(sums);
		// 向量进行水平加法，得到suml2中两元素的和以及sumh2中两元素的和
		suml2=vpadd_f32(suml2,sumh2);
		// 再次进行水平加法，得到sum4向量4个元素的和
		float32_t sum=vpadds_f32(suml2);
        hidenLayerOutput[i] = sigmoid((float)sum);
    }

	//计算输出层结点的输出值,同上
	for (int i = 0; i < numNodesOutputLayer; ++i) {
        float z = 0.0;
        float32x4_t sums = vmovq_n_f32(0);
        // 内层循环使用SSE(AVX)每次处理4(8)个float数据
        for (int j = 0; j < numNodesOutputLayer - 4; j += 4) {
			float32x4_t weights128;
			float32x4_t vecs = vld1q_f32(hidenLayerOutput + j);
			vld1q_lane_f32(weights[1][j]+i, weights128, 0);
			vld1q_lane_f32(weights[1][j+1]+i, weights128, 1);
			vld1q_lane_f32(weights[1][j+2]+i, weights128, 2);
			vld1q_lane_f32(weights[1][j+3]+i, weights128, 3);
            // 乘加指令
			weights128 = vaddq_f32(vecs,weights128);
			sums = vmulq_f32(weights128, sums);
        }
        // 4个局部和相加
		float32x2_t suml2=vget_low_f32(sums);
		// 将高位两个元素保存到sumh2向量
		float32x2_t sumh2=vget_high_f32(sums);
		// 向量进行水平加法，得到suml2中两元素的和以及sumh2中两元素的和
		suml2=vpadd_f32(suml2,sumh2);
		// 再次进行水平加法，得到sum4向量4个元素的和
		float32_t sum=vpadds_f32(suml2);
        z=(float)sum;
        z += bias[1][i];
        outputLayerOutput[i] = sigmoid(z);
        outputMat[index][i] = outputLayerOutput[i];
    }


	//计算偏置更新量allDeltaBias与allDeltaWeights[1]
	/*for (int j = 0; j < numNodesOutputLayer; ++j) {
		allDeltaBias[index][1][j] = (-0.1) * (_labelVec[j] - outputLayerOutput[j]) * outputLayerOutput[j]
			* (1 - outputLayerOutput[j]);
		for (int i = 0; i < numNodesHiddenLayer; ++i) {
			allDeltaWeights[index][1][i][j] = allDeltaBias[index][1][j] * hidenLayerOutput[i];
		}
	}*/
//printf("0");
    for (int j = numNodesOutputLayer -4;j>=0;j-= 4) {
        float32x4_t t1, t2, t3, t4;
        //t1, t2, t3, t4分别为4项乘数，每次处理4个数
        t1 = vmovq_n_f32(-0.1);
        t2 = vld1q_f32(_labelVec + j);
        t3 = vld1q_f32(outputLayerOutput + j);
        t2 = vsubq_f32(t2, t3);
        t4 = vmovq_n_f32(1);
        t4 = vsubq_f32(t4, t3);

        //t1=t1*t2*t3*t4
        t1 = vmulq_f32(t1, t2);
        t3 = vmulq_f32(t3, t4);
        t1 = vmulq_f32(t1, t3);
		vst1q_f32(allDeltaBias[index][1] + j, t1);
	}
    for (int i = numNodesHiddenLayer-1; i>= 0; i --) {
        for (int j = numNodesOutputLayer -4;j>=0;j-= 4) {
            float32x4_t a1, a2,pro;
            //标量load
            a1 = vmovq_n_f32 (hidenLayerOutput[i]);
            //向量乘
            a2 = vld1q_f32(allDeltaBias[index][1]+j);
            pro = vmulq_f32(a2, a1);
            //存储
            vst1q_f32(allDeltaWeights[index][1][i] + j, pro);
        }
    }
	   float *z=new float[numNodesHiddenLayer];
	for (int j = numNodesHiddenLayer-1; j >= 0; j --) {
        float32x4_t t1, t2, sum;
        //初始置0
        sum = vmovq_n_f32(0);
		for (int i = numNodesOutputLayer - 4; i >= 0; i -= 4){
            t1 = vld1q_f32(weights[1][j] + i);
            t2 = vld1q_f32(allDeltaBias[index][1] + i);
            //向量乘
            t1 = vmulq_f32(t1, t2);
            //向量加
            sum = vaddq_f32(sum, t1);
		}
		//横向加，最终sum中每32位均为结果。
		//横向加，最终sum中每32位均为结果。
		float32x2_t suml2=vget_low_f32(sum);
		// 将高位两个元素保存到sumh2向量
		float32x2_t sumh2=vget_high_f32(sum);
		// 向量进行水平加法，得到suml2中两元素的和以及sumh2中两元素的和
		suml2=vpadd_f32(suml2,sumh2);
		// 再次进行水平加法，得到sum4向量4个元素的和
		float32_t sum2=vpadds_f32(suml2);
		z[j]=(float)sum2;
	}
    for (int j = numNodesHiddenLayer-4; j >= 0; j -=4) {
        float32x4_t t1, t2,t3, product;
        t1 = vld1q_f32(z+j);
        t2 = vld1q_f32(hidenLayerOutput+j);
        t3 = vmovq_n_f32(1);
        t3 = vsubq_f32(t3,t2);
        product = vmulq_f32(t1,t2);
        product = vmulq_f32(product,t3);
        vst1q_f32(allDeltaBias[index][0]+j,product);
    }
    delete[]z;

	 for (int i = numNodesInputLayer-1; i>= 0; i --) {
        for (int j = numNodesHiddenLayer -4;j>=0;j-= 4) {
            float32x4_t a1, a2,pro;
            //标量load
            a1 = vmovq_n_f32 (_trainVec[i]);
            //向量乘
            a2 = vld1q_f32(allDeltaBias[index][0]+j);
            pro =  vmulq_f32(a2, a1);
            //存储
            vst1q_f32(allDeltaWeights[index][0][i] + j, pro);
        }
    }
}



bool ANN_SIMD::isNotConver_(const int _sampleNum,float** _labelMat, float _thresh)
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

	static int tt = 0;
	tt++;
	printf("第%d次训练:", tt);
	printf("%0.12f\n", lossFunc);
	
	if (lossFunc > _thresh)
		return true;

	return false;
}

void ANN_SIMD::predict(float* in, float* proba)
{
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
