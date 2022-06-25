#ifndef LAYER_H
#define LAYER_H

#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include"global.h"

class Layer
{

public:
    Layer(int, int, activationType);
    virtual ~Layer();
    void _forward(float*);
    void display();

protected:
    float** weights,** weights_delta;//为优化访问次序，weights为[输出层维度][输入层维度]
    float* bias;//[输出层维度]
    int num_input, num_output;
    float* output_nodes;//[输出层维度]
    float* bias_delta;//[输出层维度]
    float activation_function(float);
    float derivative_activation_function(float);
    friend class ANN_4;
    activationType activation_type;//可指定激活函数
};

#endif // LAYER_H
