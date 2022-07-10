#ifndef ANN_4_H
#define ANN_4_H

#include"Layer.h"
#include"global.h"
//#include <pthread.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <windows.h>
#include <ctime>
#include <math.h>
#include<vector>
#include<algorithm>
using namespace std;
class ANN_4
{
    //ANN改进版本2，重构了框架，可以指定梯度下降的batch_size
public:
    ANN_4(int* _num_each_layer, int _num_epoch = 10, int _batch_size = NUM_SAMPLE, int _num_layers = 1, float _study_rate = 0.1);
    ~ANN_4();

    void shuffle(const int num_sample, float** _trainMat, float** _labelMat);
    void train(int _sampleNum, float** _trainMat, float** _labelMat);
    void train_SIMD(const int num_sample, float** _trainMat, float** _labelMat);
    void show_predictions(float* X);
    void get_results(const int num_sample, float** _trainMat, float** _labelMat);
    void display();

private:
    int num_layers;           //网络隐藏层数，默认为1
    int* num_each_layer;      //各层维度数，[0]为输入层，[numLayers]为输出层，[1]至[numLayers-1]为隐藏层

    int batch_size;
    int num_epoch;
    vector<Layer*> layers;
    float study_rate;               //学习速率
    void predict(float* in);
    //float sigmoid(float x) { return 1 / (1 + exp(-1 * x)); }
    //float Dsigmoid(float x) { return sigmoid(x)*(1-sigmoid(x)); }
    bool isNotConver_(const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh);
    friend class Layer;

};

#endif // ANN_3_H
