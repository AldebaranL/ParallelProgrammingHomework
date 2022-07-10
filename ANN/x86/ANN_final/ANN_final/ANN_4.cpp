#include "ANN_4.h"

ANN_4::ANN_4(int* _num_each_layer, int _num_epoch, int _batch_size, int _num_layers, float _study_rate)
{
    num_layers = _num_layers;
    study_rate = _study_rate;
    num_each_layer = new int[_num_layers + 2];
    num_each_layer[0] = _num_each_layer[0];
    for (int i = 1; i <= num_layers + 1; i++)
    {
        num_each_layer[i] = _num_each_layer[i];
        layers.push_back(new Layer(num_each_layer[i - 1], num_each_layer[i],_tanh));
    }
    //display();
    num_epoch = _num_epoch;
    batch_size = _batch_size;
}

ANN_4::~ANN_4()
{
    //printf ("begin free ANN_4");
    delete[]num_each_layer;
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    // printf ("free ANN_4\n");
}
void ANN_4::shuffle(const int num_sample, float** _trainMat, float** _labelMat)
{
    //init
    int* shuffle_index = new int[num_sample];
    float** trainMat_old = new float* [num_sample];
    float** labelMat_old = new float* [num_sample];
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
    random_shuffle(shuffle_index, shuffle_index + num_sample);
    for (int i = 0; i < num_sample; i++)
    {
        for (int j = 0; j < num_each_layer[0]; j++) _trainMat[i][j] = trainMat_old[shuffle_index[i]][j];
        for (int j = 0; j < num_each_layer[num_layers + 1]; j++) _labelMat[i][j] = labelMat_old[shuffle_index[i]][j];
    }

    //delete
    for (int i = 0; i < num_sample; i++)
    {
        delete[]trainMat_old[i];
        delete[]labelMat_old[i];
    }
    delete[]trainMat_old;
    delete[]labelMat_old;
    delete[] shuffle_index;

    printf("finish shuffle\n");
}
void ANN_4::get_results(const int num_sample, float** _trainMat, float** _labelMat) {
    float tp = 0, fp = 0, tn = 0, fn = 0;
    float *Y;
    for (int index = 0; index < num_sample; index++) {
        this->predict(_trainMat[index]);
        Y = layers[num_layers]->output_nodes;
        float maxn = -1,maxi = 0;
        for (int j = 0; j < num_each_layer[num_layers + 1]; j++) {
            if (Y[j] > maxn) {
                maxn = Y[j];
                maxi = j;
            }
        }
        if (_labelMat[index][(int)maxi] == 1) {
            tp++;
        }
        else {
            fp++;
            cout << maxn << maxi;
            this->show_predictions(_trainMat[index]);
        }
    }
    cout << "tp:" << tp  << endl;
    cout << "fp:" << fp << endl;
    cout << "accuracy:" << tp / (tp + fp) << endl;
}

void ANN_4::train(const int num_sample, float** _trainMat, float** _labelMat)
{
    std::printf("begin training\n");
    float thre = 1e-2;

    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        isNotConver_(num_sample, _trainMat, _labelMat, thre);
        // if (epoch % 50 == 0) printf ("round%d:\n", epoch);
        for (int batch_i = 0; batch_i < (num_sample + batch_size - 1) / batch_size; batch_i++) {
            for (int i = num_layers; i >= 0; i--)
                for (int j = 0; j < num_each_layer[i + 1]; j++) {
                    for (int k = 0; k < num_each_layer[i]; k++) {
                        layers[i]->weights_delta[j][k] = 0;
                    }
                    layers[i]->bias_delta[j] = 0;
                }
            int t_batch_size = min((batch_i + 1) * batch_size, num_sample) - batch_i * batch_size;
            if (t_batch_size == 0)cout << "t_batch_szie wrong!!" << endl;
            for (int index = batch_i * batch_size; index < batch_i * batch_size + t_batch_size; index++) {
                //前向传播
                float output_node;
                for (int i = 0; i < num_each_layer[1]; i++)
                {
                    output_node = 0.0;
                    for (int j = 0; j < num_each_layer[0]; j++)
                    {
                        output_node += layers[0]->weights[i][j] * _trainMat[index][j];
                    }
                    output_node += layers[0]->bias[i];
                    layers[0]->output_nodes[i] = layers[0]->activation_function(output_node);//输出-1到1
                }
                for (int i_layer = 1; i_layer <= num_layers; i_layer++)
                {
                    for (int i = 0; i < num_each_layer[i_layer + 1]; i++)
                    {
                        output_node = 0.0;
                        for (int j = 0; j < num_each_layer[i_layer]; j++)
                        {
                            output_node += layers[i_layer]->weights[i][j] * layers[i_layer - 1]->output_nodes[j];
                        }
                        output_node += layers[i_layer]->bias[i];
                        layers[i_layer]->output_nodes[i] = layers[i_layer]->activation_function(output_node);
                    }
                }
                //计算delta
                float* temp_bias_delta = new float[MAX_NUM_LAYER];
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    //均方误差损失函数
                    temp_bias_delta[j] = (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function(layers[num_layers]->output_nodes[j]); //输出0到1
                    layers[num_layers]->bias_delta[j] += temp_bias_delta[j];
                    //交叉熵损失函数
                    //layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
                    for (int k = 0; k < num_each_layer[num_layers]; k++)
                        layers[num_layers]->weights_delta[j][k] += temp_bias_delta[j] * layers[num_layers - 1]->output_nodes[k];
                }
                //计算每层的delta,行访问优化
                for (int i = num_layers - 1; i >= 0; i--)
                {
                    float* error = new float[num_each_layer[i + 1]];
                    
                    for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
                    for (int k = 0; k < num_each_layer[i + 2]; k++)
                    {
                        for (int j = 0; j < num_each_layer[i + 1]; j++)
                        {
                            error[j] += layers[i + 1]->weights[k][j] * temp_bias_delta[k];
                        }
                    }
                    for (int j = 0; j < num_each_layer[i + 1]; j++)
                    {
                        temp_bias_delta[j] = error[j] * layers[num_layers]->derivative_activation_function(layers[i]->output_nodes[j]);
                        layers[i]->bias_delta[j] += temp_bias_delta[j];
                        for (int k = 0; k < num_each_layer[i]; k++)
                            if (i == 0) layers[0]->weights_delta[j][k] += temp_bias_delta[j] * _trainMat[index][k];
                            else layers[i]->weights_delta[j][k] += temp_bias_delta[j] * layers[i - 1]->output_nodes[k];
                    }
                    delete[]error;
                }

                delete[]temp_bias_delta;
            }

            //反向传播，weights和bias更新
            for (int i = 0; i <= num_layers; i++)
            {
                for (int k = 0; k < num_each_layer[i + 1]; k++)
                {
                    for (int j = 0; j < num_each_layer[i]; j++)
                    {
                        layers[i]->weights[k][j] -= study_rate * layers[i]->weights_delta[k][j] / t_batch_size;
                    }
                    layers[i]->bias[k] -= study_rate * layers[i]->bias_delta[k] / t_batch_size;
                }
            }
        }
    }
    std::printf("finish training\n");
}

bool ANN_4::isNotConver_(const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh)
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

void ANN_4::predict(float* in)
{
    layers[0]->_forward(in);
    for (int i = 1; i <= num_layers; i++)
    {
        layers[i]->_forward(layers[i - 1]->output_nodes);
    }
}
void ANN_4::display()
{
    for (int i = 0; i <= num_layers; i++)
    {
        layers[i]->display();
    }
}


void ANN_4::show_predictions(float* X)
{
    predict(X);
    static int t = 0;
    printf("in%d:", t);
    for (int i = 0; i < min(5, num_each_layer[0]); i++) printf("%f,", X[i]);
    printf("  out%d:", t);
    for (int i = 0; i < min(5, num_each_layer[num_layers + 1]); i++)
        printf("%f,", layers[num_layers]->output_nodes[i]);
    t++;
    printf("\n");
    //return layers[num_layers]->output_nodes;
}
