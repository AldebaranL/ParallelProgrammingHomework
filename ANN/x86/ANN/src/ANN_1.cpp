#include "ANN_1.h"

ANN_1::ANN_1 ( int* _num_each_layer, int _num_layers, float _study_rate)
{
    num_layers = _num_layers;
    study_rate = _study_rate;
    num_each_layer = new int[_num_layers + 2];
    num_each_layer[0] = _num_each_layer[0];
    for (int i = 1; i <= num_layers + 1; i++)
    {
        num_each_layer[i] = _num_each_layer[i];
        layers.push_back (new Layer (num_each_layer[i - 1], (num_each_layer[i]) ) );
    }
    display();
}

ANN_1::~ANN_1()
{
    delete[]num_each_layer;
    for (int i = 0; i < layers.size(); i++) delete layers[i];
}
/*




//线程函数定义

void* threadFunc(void *param) {
    hreadParam_t *p = (threadParam_t*)param;
    int t_id = p −> t_id;

    pthread_exit(NULL);
}*/

void ANN_1::train (const int num_sample, float** _trainMat, float** _labelMat)
{
    printf ("begin training\n");
    float thre = 1e-2;
    int num_epoch = 100;
    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        if(epoch%50==0)printf ("round%d:\n", epoch);
        for (int index = 0; index < num_sample; ++index)  //采用随机梯度下降法，每次使用全部样本训练
        {
            //反向传播
            back_propagation (_trainMat[index], _labelMat[index]);
        }
        //if(!isNotConver_(_sampleNum, _labelMat, thre)) break;
       // display();
    }

    //销毁所有信号量
    // sem_destroy();
    printf ("finish training\n");
}

bool ANN_1::isNotConver_ (const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh)
{
    float lossFunc = 0.0;
    for (int k = 0; k < _sampleNum; ++k)
    {
        predict(_trainMat[k]);
        float loss = 0.0;
        for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
        {
            loss += (layers[num_layers]->output_nodes[t] - _labelMat[k][t]) * (layers[num_layers]->output_nodes[t] - _labelMat[k][t]);
        }
        lossFunc += (1.0 / 2) * loss;
    }

    lossFunc = lossFunc / _sampleNum;

    static int tt = 0;
    tt++;
    printf ("第%d次训练：%0.12f\n", tt, lossFunc);

    if (lossFunc > _thresh) return true;
    return false;
}

void ANN_1::predict (float* in)
{
    layers[0]->_forward (in);
    for (int i = 1; i <= num_layers; i++)
    {
        layers[i]->_forward (layers[i - 1]->output_nodes);
    }
}
void ANN_1::display(){
    for (int i = 0; i <= num_layers; i++)
    {
        layers[i]->display();
    }
}
void ANN_1::back_propagation (float* X, float * Y)
{
    //前向传播
    predict (X);
    //printf ("finish predict\n");

    //计算error和delta
    for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
    {
        layers[num_layers]->error[j] = layers[num_layers]->output_nodes[j] - Y[j];
        layers[num_layers]->delta[j] = layers[num_layers]->error[j] *
                                       layers[num_layers]->derivative_activation_function (layers[num_layers]->output_nodes[j]);
    }
    for (int i = num_layers - 1; i >= 0; i--)
    {
        for (int j = 0; j < num_each_layer[i + 1]; j++)
        {
            layers[i]->error[j] = 0.0;
            for (int k = 0; k < num_each_layer[i + 2]; k++)
            {
                layers[i]->error[j] += layers[i + 1]->weights[k][j] * layers[i + 1]->delta[k];
            }
            layers[i]->delta[j] = layers[i]->error[j] * layers[num_layers]->derivative_activation_function (layers[i]->output_nodes[j]);
        }
    }
    //printf ("finish cal error\n");

    //weights和bais更新
    for (int k = 0; k < num_each_layer[1]; k++)
    {
        for (int j = 0; j < num_each_layer[0]; j++)
        {
            layers[0]->weights[k][j] -= study_rate * X[j] * layers[0]->delta[k];
        }
        layers[0]->bias[k] -=  study_rate * layers[0]->delta[k];
    }
    for (int i = 1; i <= num_layers; i++)
    {
        for (int k = 0; k < num_each_layer[i + 1]; k++)
        {
            for (int j = 0; j < num_each_layer[i]; j++)
            {
                layers[i]->weights[k][j] -= study_rate * layers[i - 1]->output_nodes[j] * layers[i]->delta[k];
            }
            layers[i]->bias[k] -=  study_rate * layers[i]->delta[k];
        }
    }

}
void ANN_1::get_predictions (float* X)
{
    for (int i = 0; i < num_each_layer[0]; i++) printf ("%f,", X[i]);
    predict (X);
    static int t = 0;
    printf ("in%d:", t);
    for (int i = 0; i < num_each_layer[0]; i++) printf ("%f,", X[i]);
    printf ("  out%d:", t);
    for (int i = 0; i < num_each_layer[num_layers + 1]; i++)
        printf ("%f,", layers[num_layers]->output_nodes[i]);
    t++;
    printf ("\n");
}
