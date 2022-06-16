#ifndef LAYER_H
#define LAYER_H

#include<stdlib.h>
#include<math.h>
#include<stdio.h>
class Layer
{

public:
    Layer (int, int);
    virtual ~Layer();
    void _forward (float *);
    void display();

protected:
    float **weights;//为优化访问次序，weights为[输出层维度][输入层维度]
    float *bias;//[输出层维度]
    int num_input, num_output;
    float *output_nodes;
    float *error;//在ANN_2及以后不再使用
    float *delta;//[输出层维度]
    float activation_function (float);
    float derivative_activation_function (float);
    friend class ANN_pthread;
    friend class ANN_new;
    friend class ANN_1;
    friend class ANN_2;
    friend class ANN_3;
    friend class ANN_openMP;
    friend class ANN_MPI;
    enum {sigmoid,tanh} activation_type;//可指定激活函数
};

#endif // LAYER_H
