#ifndef ANN_1_H
#define ANN_1_H

#include<Layer.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <windows.h>
#include <ctime>
#include <math.h>
#include<vector>
using namespace std;
class ANN_1
{
    //ANN改进版本1，重构了框架，可以添加任意多隐藏层，仅可以使用随机梯度下降
public:
    ANN_1 ( int* _num_each_layer, int _num_layers = 1, float _study_rate = 0.1);
    ~ANN_1();

    void train (int _sampleNum, float** _trainMat, float** _labelMat);
    void get_predictions (float* X);
    void display();

private:
    int num_layers;           //网络隐藏层数，默认为1
    int* num_each_layer;            //各层维度数[0]为输入层，[numLayers]为输出层，[1]至[numLayers-1]为隐藏层

    vector<Layer*> layers;
    float study_rate;               //学习速率
    void back_propagation (float* X, float * Y);
    void predict (float* in);

    //float sigmoid(float x) { return 1 / (1 + exp(-1 * x)); }
    //float Dsigmoid(float x) { return sigmoid(x)*(1-sigmoid(x)); }
    bool isNotConver_ (const int _sampleNum,float** _trainMat, float** _labelMat, float _thresh);
    friend class Layer;

    //线程数据结构定义
    typedef struct {
        int t_id; //线程 id
    }threadParam_t;
    //信号量定义
    //sem_t sem_leader
    //sem_t sem_Divsion[NUM_THREADS−1];
    //sem_t sem_Elimination[NUM_THREADS−1];

    void* threadFunc(void *param) ;
};

#endif // ANN_1_H
