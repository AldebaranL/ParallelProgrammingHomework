#include "Layer.h"

Layer::Layer(int _num_input, int _num_output, activationType  _activation_type=_tanh)
{
    //ctor
    num_input = _num_input;
    num_output = _num_output;
    weights = new float* [num_output];
    for (int i = 0; i < num_output; i++) weights[i] = new float[num_input];
    weights_delta = new float* [num_output];
    for (int i = 0; i < num_output; i++) weights_delta[i] = new float[num_input];

    bias = new float[num_output];
    output_nodes = new float[num_output];
    //error = new float[num_output];
    bias_delta = new float[num_output];

    for (int i = 0; i < num_output; i++)
    {
        for (int j = 0; j < num_input; j++)
        {
            weights[i][j] = ((rand() % 20000) - 10000) / 10000.0;
        }
        bias[i] = ((rand() % 20000) - 10000) / 10000.0;
    }
    //activation_type = tanh;
    activation_type = _activation_type;
}

Layer::~Layer()
{
    //dtor
    delete[] bias;
    delete[] output_nodes;
    for (int i = 0; i < num_output; i++) {
        delete[] weights[i];
        delete[] weights_delta[i];
    }
    delete[]weights_delta;
   // delete[]error;
    delete[]bias_delta;
}
void Layer::display()
{
    printf("%d,%d:\n", num_input, num_output);
    for (int i = 0; i < num_input; i++)
    {
        for (int j = 0; j < num_output; j++)
        {
            printf("%f ", weights[j][i]);
        }
        printf("\n");
    }
}
void Layer::_forward(float* input_nodes)
{
    for (int i = 0; i < num_output; i++)
    {
        output_nodes[i] = 0.0;
        for (int j = 0; j < num_input; j++)
        {
            output_nodes[i] += weights[i][j] * input_nodes[j];
        }
        output_nodes[i] += bias[i];
        output_nodes[i] = activation_function(output_nodes[i]);
        if (output_nodes[i] == NAN)printf("!!!!!!!");
    }
}
float Layer::activation_function(float x)
{

    switch (activation_type)
    {
    case _sigmoid:
        return 1 / (1 + exp(-1 * x));
    case _tanh:
        return (exp(x) - exp(-1 * x)) / (exp(x) + exp(-1 * x));
    default:
        return x;
    }
}
float Layer::derivative_activation_function(float x)
{
    switch (activation_type)
    {
    case _sigmoid:
        return activation_function(x) * (1 - activation_function(x));
        // return x*(1-x);
    case _tanh:
        return 1 - activation_function(x) * activation_function(x);
    default:
        return x;
    }
}
