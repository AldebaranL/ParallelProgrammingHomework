#include "ANN_2.h"

ANN_2::ANN_2 ( int* _num_each_layer, int _num_epoch, int _batch_size, int _num_layers, float _study_rate)
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

ANN_2::~ANN_2()
{
    //printf ("begin free ANN_2");
    delete[]num_each_layer;
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    // printf ("free ANN_2\n");
}
void ANN_2::shuffle (const int num_sample, float** _trainMat, float** _labelMat)
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

void ANN_2::train_SIMD  (const int num_sample, float** _trainMat, float** _labelMat)
{
    printf ("begin training\n");
    float thre = 1e-2;
    float *avr_X = new float[num_each_layer[0]];

    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        //if (epoch % 5 == 0) printf ("round%d:\n", epoch);
        int index = 0;

        ANN_2 *class_p = this;
        while (index < num_sample)
        {
            for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] = 0.0;
            for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
            {
                layers[num_layers]->delta[j] = 0.0;
            }

            for (int i = 0; i < batch_size; i++, index++) //默认batch_size=1，即采用随机梯度下降法，每次使用全部样本训练并更新参数
            {
                //前向传播

                for (int i = 0; i < num_each_layer[1]; i++)
                {
                    layers[0]->output_nodes[i] = 0.0;
                    //printf("%d,%d ",i,class_p->num_each_layer[1]);
                    int max_j = class_p->num_each_layer[0];
                    float32x4_t ans = vmovq_n_f32(0);
                    for (int j = 0; j < max_j; j += 4)
                        // for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                    {
                        float32x4_t t1, t2;
                        //将内部循环改为4位的向量运算
                        t1 = vld1q_f32 (layers[0]->weights[i] + j);
                        t2 = vld1q_f32 (_trainMat[index] + j);
                        t1 = vmulq_f32 (t1, t2);

                        ans = vaddq_f32 (ans, t1);

                        // printf("%d,%d ",j,class_p->num_each_layer[0]);
                        //  class_p->layers[0]->output_nodes[i] += class_p->layers[0]->weights[i][j] * p->sampleMat[sample_index][j];
                        // printf("%d,%d ",i,sample_index);
                    }
                            // 4个局部和相加
		float32x2_t suml2=vget_low_f32(ans);
		// 将高位两个元素保存到sumh2向量
		float32x2_t sumh2=vget_high_f32(ans);
		// 向量进行水平加法，得到suml2中两元素的和以及sumh2中两元素的和
		suml2=vpadd_f32(suml2,sumh2);
		// 再次进行水平加法，得到sum4向量4个元素的和
		float32_t sum=vpadds_f32(suml2);
                    layers[0]->output_nodes[i] += sum;
                    layers[0]->output_nodes[i] += layers[0]->bias[i];
                    layers[0]->output_nodes[i] = layers[0]->activation_function (layers[0]->output_nodes[i]);
                }
                // printf("@");
                for (int layers_i = 1; layers_i <= num_layers; layers_i++)
                {

                    for (int i = 0; i < num_each_layer[layers_i + 1]; i++)
                    {
                        class_p->layers[layers_i]->output_nodes[i] = 0.0;
                        int max_j = class_p->num_each_layer[layers_i];
                        float32x4_t ans = vmovq_n_f32(0);
                        for (int j = 0; j < max_j; j += 4)
                            // for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                        {
                            float32x4_t t1, t2;
                            //将内部循环改为4位的向量运算
                            t1 = vld1q_f32 (class_p->layers[layers_i]->weights[i] + j);
                            t2 = vld1q_f32 (_trainMat[index] + j);
                            t1 = vmulq_f32 (t1, t2);

                            ans = vaddq_f32 (ans, t1);
                        }
                        // 4个局部和相加
                            // 4个局部和相加
		float32x2_t suml2=vget_low_f32(ans);
		// 将高位两个元素保存到sumh2向量
		float32x2_t sumh2=vget_high_f32(ans);
		// 向量进行水平加法，得到suml2中两元素的和以及sumh2中两元素的和
		suml2=vpadd_f32(suml2,sumh2);
		// 再次进行水平加法，得到sum4向量4个元素的和
		float32_t sum=vpadds_f32(suml2);
                        class_p->layers[layers_i]->output_nodes[i] += sum;
                        class_p->layers[layers_i]->output_nodes[i] += class_p->layers[layers_i]->bias[i];
                        class_p->layers[layers_i]->output_nodes[i] = class_p->layers[layers_i]->activation_function (class_p->layers[layers_i]->output_nodes[i]);
                    }
                }

                static int tt = 0;
                float loss = 0.0;
                for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
                {
                    loss += (layers[num_layers]->output_nodes[t] - _labelMat[index][t]) * (layers[num_layers]->output_nodes[t] - _labelMat[index][t]);
                }
                //printf ("第%d次训练：%0.12f\n", tt++,loss);


                //for (int i = 0; i < min (5, num_each_layer[num_layers + 1]); i++)
                //printf ("%f,", layers[num_layers]->output_nodes[i]);
                //printf ("\n");
                // printf ("finish predict\n");

                //计算loss，即最后一层的delta,即该minibatch中所有loss的平均值
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    //均方误差损失函数
                    layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function (layers[num_layers]->output_nodes[j]);
                    //交叉熵损失函数
                    //layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
                }
                //printf ("finish cal error\n");
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] += _labelMat[index][X_i];
            }
            if (index % batch_size == 0)
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= batch_size;
                    //for(int i=0;i<5;i++)printf("delta=%f\n",layers[num_layers]->delta[i]);
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= batch_size;

            }
            else
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (index % batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= (index % batch_size);
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= (index % batch_size);
            }

            //反向传播更新参数
            for (int k = 0; k < class_p->num_each_layer[1]; k++)
            {
                int max_j = class_p->num_each_layer[0];
                float32x4_t sr = vmovq_n_f32 (class_p->study_rate);
                float32x4_t t2 = vmovq_n_f32 (class_p->layers[0]->delta[k]);
                for (int j = 0; j < max_j; j += 4)

                    //for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                {
                    float32x4_t t1, t3, product;
                    t1 = vld1q_f32 (class_p->layers[0]->weights[k] + j);
                    t3 = vld1q_f32 (avr_X + j);
                    //product=sr*t2*t3，向量对位相乘
                    product = vmulq_f32 (t3, t2);
                    product = vmulq_f32 (product, sr);
                    t1 = vsubq_f32 (t1, product);
                    vst1q_f32 (class_p->layers[0]->weights[k] + j, t1);

                    //class_p->layers[0]->weights[k][j] -= class_p->study_rate * p->sampleMat[sample_index - 1][j] * class_p->layers[0]->delta[k];
                }
                class_p->layers[0]->bias[k] -=  class_p->study_rate *  class_p->layers[0]->delta[k];
            }
            for (int i = 1; i <= class_p->num_layers; i++)
            {
                for (int k = 0; k < class_p->num_each_layer[i + 1]; k++)
                {
                    int max_j = class_p->num_each_layer[i];
                    float32x4_t sr = vmovq_n_f32 (class_p->study_rate);
                    float32x4_t t2 = vmovq_n_f32 (class_p->layers[i]->delta[k]);//printf("1");
                    for (int j = 0; j < max_j; j += 4)
                        //for (int j = t_id; j < class_p->num_each_layer[i]; j += NUM_THREADS)
                    {
                        float32x4_t t1, t3, product;
                        t1 = vld1q_f32 (class_p->layers[i]->weights[k] + j);
                        t3 = vld1q_f32 (class_p->layers[i - 1]->output_nodes + j);
                        //product=sr*t2*t3，向量对位相乘
                        product = vmulq_f32 (t3, t2);
                        product = vmulq_f32 (product, sr);
                        t1 = vsubq_f32 (t1, product);

                        vst1q_f32 (class_p->layers[i]->weights[k] + j, t1);

                        //class_p->layers[i]->weights[k][j] -= class_p->study_rate * class_p->layers[i - 1]->output_nodes[j] * class_p->layers[i]->delta[k];
                    }//printf("2");
                    class_p->layers[i]->bias[k] -=  class_p->study_rate * class_p->layers[i]->delta[k];
                }
            }

            //printf ("finish bp with index:%d\n",index);
        }



        // display();
    }
    printf ("finish training\n");
    delete[]avr_X;
}
void ANN_2::train (const int num_sample, float** _trainMat, float** _labelMat)
{
    printf ("begin training\n");
    float thre = 1e-2;
    float *avr_X = new float[num_each_layer[0]];

    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        //if (epoch % 50 == 0) printf ("round%d:\n", epoch);
        int index = 0;


        while (index < num_sample)
        {
            for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] = 0.0;
            for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
            {
                layers[num_layers]->delta[j] = 0.0;
            }

            for (int i = 0; i < batch_size; i++, index++) //默认batch_size=1，即采用随机梯度下降法，每次使用全部样本训练并更新参数
            {
                //前向传播
                predict (_trainMat[index]);
                static int tt = 0;
                float loss = 0.0;
                for (int t = 0; t < num_each_layer[num_layers + 1]; ++t)
                {
                    loss += (layers[num_layers]->output_nodes[t] - _labelMat[index][t]) * (layers[num_layers]->output_nodes[t] - _labelMat[index][t]);
                }
                //printf ("第%d次训练：%0.12f\n", tt++,loss);


                //for (int i = 0; i < min (5, num_each_layer[num_layers + 1]); i++)
                //printf ("%f,", layers[num_layers]->output_nodes[i]);
                //printf ("\n");
                // printf ("finish predict\n");

                //计算loss，即最后一层的delta,即该minibatch中所有loss的平均值
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    //均方误差损失函数
                    layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]) * layers[num_layers]->derivative_activation_function (layers[num_layers]->output_nodes[j]);
                    //交叉熵损失函数
                    //layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[index][j]);
                }
                //printf ("finish cal error\n");
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] += _labelMat[index][X_i];
            }
            if (index % batch_size == 0)
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= batch_size;
                    //for(int i=0;i<5;i++)printf("delta=%f\n",layers[num_layers]->delta[i]);
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= batch_size;

            }
            else
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    if (index % batch_size == 0) printf ("wrong!\n");
                    layers[num_layers]->delta[j] /= (index % batch_size);
                }
                for (int X_i = 0; X_i < num_each_layer[0]; X_i++) avr_X[X_i] /= (index % batch_size);
            }

            //反向传播更新参数
            back_propagation (avr_X, _labelMat[index - 1]);
            //printf ("finish bp with index:%d\n",index);
        }



        // display();
    }
    printf ("finish training\n");
    delete[]avr_X;
}

bool ANN_2::isNotConver_ (const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh)
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
    printf ("第%d次训练：%0.12f\n", tt, lossFunc);

    if (lossFunc > _thresh) return true;
    return false;
}

void ANN_2::predict (float* in)
{
    layers[0]->_forward (in);
    for (int i = 1; i <= num_layers; i++)
    {
        layers[i]->_forward (layers[i - 1]->output_nodes);
    }
}
void ANN_2::display()
{
    for (int i = 0; i <= num_layers; i++)
    {
        layers[i]->display();
    }
}

void ANN_2::back_propagation (float* X, float * Y)
{
    //计算每层的delta,行访问优化
    for (int i = num_layers - 1; i >= 0; i--)
    {
        float *error = new float[num_each_layer[i + 1]];
        for (int j = 0; j < num_each_layer[i + 1]; j++) error[j] = 0.0;
        for (int k = 0; k < num_each_layer[i + 2]; k++)
        {
            for (int j = 0; j < num_each_layer[i + 1]; j++)
            {
                error[j] += layers[i + 1]->weights[k][j] * layers[i + 1]->delta[k];
            }
        }
        for (int j = 0; j < num_each_layer[i + 1]; j++)
        {
            layers[i]->delta[j] = error[j] * layers[num_layers]->derivative_activation_function (layers[i]->output_nodes[j]);
        }
        delete[]error;
    }

    //反向传播，weights和bias更新
    for (int k = 0; k < num_each_layer[1]; k++)
    {
        for (int j = 0; j < num_each_layer[0]; j++)
        {
            layers[0]->weights[k][j] -= study_rate * X[j] * layers[0]->delta[k];
        }
        layers[0]->bias[k] -= study_rate * layers[0]->delta[k];
    }
    for (int i = 1; i <= num_layers; i++)
    {
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
void ANN_2::get_predictions (float* X)
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
