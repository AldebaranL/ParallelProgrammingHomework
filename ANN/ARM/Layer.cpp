#include "Layer.h"

Layer::Layer (int _num_input, int _num_output)
{
    //ctor
    num_input = _num_input;
    num_output = _num_output;
    weights = new float*[num_output];
    for (int i = 0; i < num_output; i++) weights[i] = new float[num_input];
    bias = new float[num_output];
    output_nodes = new float[num_output];
    error = new float[num_output];
    delta = new float[num_output];

    for (int i = 0; i < num_output; i++)
    {
        for (int j = 0; j < num_input; j++)
        {
            weights[i][j] = ( (rand() % 2000) - 1000) / 1000.0;
        }
        bias[i] = ( (rand() % 2000) - 1000) / 1000.0;
    }
    //activation_type = tanh;
    activation_type=sigmoid;
}

Layer::~Layer()
{
    //dtor
    delete[] bias;
    delete[] output_nodes;
    for (int i = 0; i < num_output; i++) delete[] weights[i];
    delete[]weights;
    delete[]error;
    delete[]delta;
}
void Layer::display ()
{
    printf ("%d,%d:\n", num_input, num_output);
    for (int i = 0; i < num_input; i++)
    {
        for (int j = 0; j < num_output; j++)
        {
            printf ("%f ", weights[j][i]);
        }
        printf ("\n");
    }
}
void Layer::_forward (float * input_nodes)
{
    for (int i = 0; i < num_output; i++)
    {
        output_nodes[i] = 0.0;
        for (int j = 0; j < num_input; j++)
        {
            output_nodes[i] += weights[i][j] * input_nodes[j];
        }
        output_nodes[i] += bias[i];
        output_nodes[i] = activation_function (output_nodes[i]);
        if(output_nodes[i]==NAN)printf("!!!!!!!");
    }
}
float Layer::activation_function (float x)
{
    switch (activation_type)
    {
    case sigmoid:
        return 1 / (1 + exp (-1 * x) );
    case tanh:
        return (exp (x) - exp (-1 * x) ) / (exp (x) + exp (-1 * x) );
    }
}
float Layer::derivative_activation_function (float x)
{
    switch (activation_type)
    {
    case sigmoid:
        return activation_function (x) * (1 - activation_function (x) );
    case tanh:
        return 1 - activation_function (x) * activation_function (x);
    }
}
