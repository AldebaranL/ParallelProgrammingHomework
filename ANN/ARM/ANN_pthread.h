#ifndef ANN_pthread_H
#define ANN_pthread_H

#include "Layer.h"
#include <pthread.h>
#include <arm_neon.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <ctime>
#include <math.h>
#include<vector>
#include<algorithm>
#include <semaphore.h>
#include"global.h"
//../../Sources/include/
using namespace std;

class ANN_pthread
{
public:
    ANN_pthread ( int* _num_each_layer, int _num_epoch = 10, int _batch_size = 1, int _num_layers = 1, float _study_rate = 0.1);
    ~ANN_pthread();

    void shuffle (const int num_sample, float** _trainMat, float** _labelMat);
    void train_semSIMD (int _sampleNum, float** _trainMat, float** _labelMat);
    void train_barrier (const int _num_sample, float** _trainMat, float** _labelMat);
	void train_sem (const int _num_sample, float** _trainMat, float** _labelMat);
    void get_predictions (float* X);
    void display();

    void* threadFunc_sem (void *param);
    void* threadFunc_sem_SIMD (void *param);
    void* threadFunc_barrier(void *param);
private:
    int num_layers;           //网络隐藏层数，默认为1
    int* num_each_layer;            //各层维度数[0]为输入层，[numLayers]为输出层，[1]至[numLayers-1]为隐藏层

    int batch_size;
    int num_epoch;
    vector<Layer*> layers;
    float study_rate;               //学习速率
    void back_propagation (float* X, float * Y);
    void predict (float* in);

    //float sigmoid(float x) { return 1 / (1 + exp(-1 * x)); }
    //float Dsigmoid(float x) { return sigmoid(x)*(1-sigmoid(x)); }
    bool isNotConver_ (const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh);
    friend class Layer;

    //int sample_index;
    //int num_sample;


    //extern sem_t *sem_before_bp;// 每个线程有自己专属的信号量
    //extern sem_t *sem_before_fw;
    //extern sem_t sem_main_after_bp;
    //extern sem_t sem_main_after_fw;



};

#endif // ANN_pthread_H
