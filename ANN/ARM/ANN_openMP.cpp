#include "ANN_openMP.h"

ANN_openMP::ANN_openMP ( int* _num_each_layer, int _num_epoch, int _batch_size, int _num_layers, float _study_rate)
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
    //display();
    num_epoch = _num_epoch;
    batch_size = _batch_size;
}

ANN_openMP::~ANN_openMP()
{
    //printf ("begin free ANN_openMP");
    delete[]num_each_layer;
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    // printf ("free ANN_openMP\n");
}
void ANN_openMP::shuffle (const int num_sample, float** _trainMat, float** _labelMat)
{
    //init
    int *shuffle_index = new int[num_sample];
    float **trainMat_old = new float*[num_sample];
    float **labelMat_old = new float*[num_sample];
    for (int i = 0; i < num_sample; i++)
    {
        trainMat_old[i] = new float[num_each_layer[0]];
        labelMat_old[i] = new float[num_each_layer[num_layers + 1]];
    }
    for (int i = 0; i < num_sample; i++)
    {
        for (int j = 0; j < num_each_layer[0]; j++)
            trainMat_old[i][j] = _trainMat[i][j];
        for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
            labelMat_old[i][j] = _labelMat[i][j];
    }

    //shuffle
    for (int i = 0; i < num_sample; i++) shuffle_index[i] = i;
    random_shuffle (shuffle_index, shuffle_index + num_sample);
    for (int i = 0; i < num_sample; i++)
    {
        for (int j = 0; j < num_each_layer[0]; j++) _trainMat[i][j] = trainMat_old[shuffle_index[i]][j];
        for (int j = 0; j < num_each_layer[num_layers + 1]; j++) _labelMat[i][j] = labelMat_old[shuffle_index[i]][j];
    }

    //delete
    for (int i = 0; i < num_sample; i++)
    {
        delete []trainMat_old[i];
        delete []labelMat_old[i];
    }
    delete []trainMat_old;
    delete []labelMat_old;
    delete[] shuffle_index;

    printf ("finish shuffle\n");
}

void ANN_openMP::train (const int num_sample, float** _trainMat, float** _labelMat)
{

    printf ("begin training\n");
    float thre = 1e-2;
    float *avr_X = new float[num_each_layer[0]];
    float *avr_Y = new float[num_each_layer[num_layers + 1]];


    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        if (epoch % 50 == 0)
        {
            //  printf ("round%d:\n", epoch);
        }
        int index = 0;

        while (index < num_sample)
        {
            for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] = 0.0;
            for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] = 0.0;

            for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
            {
                layers[num_layers]->delta[j] = 0.0;
            }

            for (int batch_i = 0; batch_i < batch_size && index < num_sample; batch_i++, index++) //默认batch_size=1，即采用随机梯度下降法，每次使用全部样本训练并更新参数
            {
                //前向传播
                #pragma omp parallel num_threads(NUM_THREADS)
                {
                    #pragma omp for
                    for (int i = 0; i < num_each_layer[1]; i++)
                    {
                        layers[0]->output_nodes[i] = 0.0;

                        for (int j = 0; j < num_each_layer[0]; j++)
                        {
                            layers[0]->output_nodes[i] += layers[0]->weights[i][j] * _trainMat[index][j];
                        }
                        layers[0]->output_nodes[i] += layers[0]->bias[i];
                        layers[0]->output_nodes[i] = layers[0]->activation_function (layers[0]->output_nodes[i]);
                    }
                }
                for (int i_layer = 1; i_layer <= num_layers; i_layer++)
                {
                    #pragma omp parallel num_threads(NUM_THREADS)
                    {
                        #pragma omp for
                        for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
                        {
                            layers[i_layer]->output_nodes[i] = 0.0;

                            for (int j = 0; j < num_each_layer[i_layer]; j++)
                            {
                                layers[i_layer]->output_nodes[i] += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
                            }
                            layers[i_layer]->output_nodes[i] += layers[i_layer]->bias[i];
                            layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function (layers[i_layer]->output_nodes[i]);

                        }

                    }

                }
//                static int tt = 0;
//                float loss = 0.0;
//                for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
//                {
//                    loss += (layers[num_layers]->output_nodes[t] - _labelMat[index][t]) * (layers[num_layers]->output_nodes[t] - _labelMat[index][t]);
//                }
//                printf ("第%d次训练：%0.12f\n", tt++,loss);


                //for (int i = 0; i < min (5, num_each_layer[num_layers + 1]); i++)
                //printf ("%f,", layers[num_layers]->output_nodes[i]);
                //printf ("\n");
                // printf ("finish predict\n");

                //计算loss，即最后一层的delta,即该minibatch中所有loss的平均值
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    //均方误差损失函数,batch内取平均
                    layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function (layers[num_layers]->output_nodes[j]);
                    //交叉熵损失函数
                    //layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
                }
                // printf ("finish cal error\n");
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] += _trainMat[index][X_i];
                for (int Y_i = 0; Y_i < num_each_layer[num_layers + 1]; Y_i++) avr_Y[Y_i] += _labelMat[index][Y_i];
            }

            //delta在batch内取平均，avr_X、avr_Y分别为本个batch的输入、输出向量的平均值
            if (index % batch_size == 0)
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= batch_size;
                    //for(int i=0;i<5;i++)printf("delta=%f\n",layers[num_layers]->delta[i]);
                    avr_Y[j] /= batch_size;
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= batch_size;

            }
            else
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (index % batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= (index % batch_size);
                    avr_Y[j] /= (index % batch_size);
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= (index % batch_size);
            }
            // printf("index:%d\n",index);
            //计算loss func，仅用于输出
            if ( index >= num_sample)
            {
                static int tt = 0;
                float loss = 0.0;
                for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
                {
                    loss += (layers[num_layers]->output_nodes[t] - avr_Y[t]) * (layers[num_layers]->output_nodes[t] - avr_Y[t]);
                }
                //  printf ("第%d次训练：%0.12f\n", tt++, loss);
            }

            //反向传播更新参数

            //计算每层的delta,行访问优化
            for (int i = num_layers - 1; i >= 0; i--)
            {
                #pragma omp parallel num_threads(NUM_THREADS)
                {
                    float *error = new float[num_each_layer[i + 1]];
//                    for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
//
//                    for (int k = 0; k < num_each_layer[i + 2]; k++)
//                    {
//                        for (int j = 0; j < num_each_layer[i + 1]; j++)
//                        {
//                            error[j] += layers[i + 1]->weights[k][j] * layers[i + 1]->delta[k];
//                        }
//                    }
//                    for (int j = 0; j < num_each_layer[i + 1]; j++)
//                    {
//                        layers[i]->delta[j] = error[j] * layers[num_layers]->derivative_activation_function (layers[i]->output_nodes[j]);
//                    }
                    #pragma omp for
                    for (int j = 0; j < num_each_layer[i + 1]; j++)
                    {
                        error[j] = 0.0;
                        for (int k = 0; k < num_each_layer[i + 2]; k++)
                        {
                            error[j] += layers[i + 1]->weights[k][j] * layers[i + 1]->delta[k];
                        }
                        layers[i]->delta[j] = error[j] * layers[num_layers]->derivative_activation_function (layers[i]->output_nodes[j]);
                    }

                    delete[]error;
                }
            }
            //反向传播，weights和bias更新
            #pragma omp parallel num_threads(NUM_THREADS)
            {
                #pragma omp for
                for (int k = 0; k < num_each_layer[1]; k++)
                {

                    for (int j = 0; j < num_each_layer[0]; j++)
                    {
                        layers[0]->weights[k][j] -= study_rate * avr_X[j] * layers[0]->delta[k];
                    }
                    layers[0]->bias[k] -= study_rate * layers[0]->delta[k];
                }
            }
            for (int i = 1; i <= num_layers; i++)
            {
                #pragma omp parallel num_threads(NUM_THREADS)
                {
                    #pragma omp for
                    for (int k = 0; k < num_each_layer[i + 1]; k++)
                    {

                        for (int j = 0; j < num_each_layer[i]; j++)
                        {
                            layers[i]->weights[k][j] -= study_rate * layers[i - 1]->output_nodes[j] * layers[i]->delta[k];
                        }
                        layers[i]->bias[k] -= study_rate * layers[i]->delta[k];
                    }
                }
            }
            //printf ("finish bp with index:%d\n",index);
        }
    }

    // display();

    printf ("finish training\n");
    delete[]avr_X;
    delete[]avr_Y;
}

bool ANN_openMP::isNotConver_ (const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh)
{
    float lossFunc = 0.0;
    for (int k = 0; k < _sampleNum; ++k)
    {
        predict (_trainMat[k]);
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
    // printf ("第%d次训练：%0.12f\n", tt, lossFunc);

    if (lossFunc > _thresh) return true;
    return false;
}

void ANN_openMP::predict (float* in)
{
    layers[0]->_forward (in);
    for (int i = 1; i <= num_layers; i++)
    {
        layers[i]->_forward (layers[i - 1]->output_nodes);
    }
}
void ANN_openMP::display()
{
    for (int i = 0; i <= num_layers; i++)
    {
        layers[i]->display();
    }
}


void ANN_openMP::get_predictions (float* X)
{
    predict (X);
    static int t = 0;
    printf ("in%d:", t);
    for (int i = 0; i < min (5, num_each_layer[0]); i++) printf ("%f,", X[i]);
    printf ("  out%d:", t);
    for (int i = 0; i < min (5, num_each_layer[num_layers + 1]); i++)
        printf ("%f,", layers[num_layers]->output_nodes[i]);
    t++;
    printf ("\n");
}
