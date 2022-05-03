#ifndef ANN_2_H
#define ANN_2_H

#include"Layer.h"
#include <arm_neon.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <ctime>
#include <math.h>
#include<vector>
#include<algorithm>
using namespace std;
class ANN_2
{
    //ANN改进版本2，重构了框架，可以指定梯度下降的batch_size
public:
    ANN_2 ( int* _num_each_layer,int _num_epoch=10,int _batch_size=1, int _num_layers = 1, float _study_rate = 0.1);
    ~ANN_2();

    void shuffle (const int num_sample, float** _trainMat, float** _labelMat);
    void train (int _sampleNum, float** _trainMat, float** _labelMat);
    void train_SIMD  (const int num_sample, float** _trainMat, float** _labelMat);
    void get_predictions (float* X);
    void display();

private:
    int num_layers;           //网络隐藏层数，默认为1
    int* num_each_layer;      //各层维度数，[0]为输入层，[numLayers]为输出层，[1]至[numLayers-1]为隐藏层

    int batch_size;
    int num_epoch;
    vector<Layer*> layers;
    float study_rate;               //学习速率
    void back_propagation (float* X, float * Y);
    void predict (float* in);

    //float sigmoid(float x) { return 1 / (1 + exp(-1 * x)); }
    //float Dsigmoid(float x) { return sigmoid(x)*(1-sigmoid(x)); }
    bool isNotConver_ (const int _sampleNum,float** _trainMat, float** _labelMat, float _thresh);
    friend class Layer;

};

#endif // ANN_2_H

