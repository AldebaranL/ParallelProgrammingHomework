#include "ANN_new.h"

ANN_new::ANN_new ( int* _num_each_layer,  int _num_epoch, int _batch_size, int _num_layers, float _study_rate)
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

ANN_new::~ANN_new()
{
    // printf("%d ",&num_each_layer);
    delete[]num_each_layer;
    // printf("%d ",layers.size());
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    // printf("free ANN_new\n");
}
void ANN_new::shuffle (const int num_sample, float** _trainMat, float** _labelMat)
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

static void *test (void * arg)
{
    return static_cast<ANN_new *> (arg)->threadFunc_sem_SIMD (arg);
}
void* ANN_new::threadFunc_sem_SIMD (void *param)
{
    threadParam_t *p = (threadParam_t*) param;
    int t_id = p -> t_id;
    ANN_new * class_p = p->class_pointer;
    //printf ("%d begin\n", t_id);
    for (int epoch = 0; epoch < class_p->num_epoch; epoch++)
    {
        int sample_index = 0;
        while (sample_index < NUM_SAMPLE)
        {
            //printf ("%d batch_size=%d,%d\n", t_id, &class_p->batch_size, class_p->batch_size);
            for (int batch_i = 0; batch_i < class_p->batch_size; batch_i++)
            {
                //printf ("%d batch_i=%d,batch_size=%d\n", t_id, batch_i, class_p->batch_size);
                if (sample_index >= NUM_SAMPLE)
                {
                    //printf ("&d,break\n", NUM_SAMPLE);
                    break;
                }
                //printf ("%d wait fw\n", t_id);
                sem_wait (& (sem_before_fw[t_id]) ); // 阻塞,等待主线程（操作自己专属的信号量）
                //printf ("%d begin fw\n", t_id);
                for (int i = 0; i < class_p->num_each_layer[1]; i++)
                {
                    class_p->layers[0]->output_nodes[i] = 0.0;
                    //printf("%d,%d ",i,class_p->num_each_layer[1]);
                    int max_j = class_p->num_each_layer[0];

                    __m128 ans = _mm_setzero_ps();
                    for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 4)
                        // for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                    {
                        __m128 t1, t2;
                        //将内部循环改为4位的向量运算
                        t1 = _mm_loadu_ps (class_p->layers[0]->weights[i] + j);
                        t2 = _mm_loadu_ps (p->sampleMat[sample_index] + j);
                        t1 = _mm_mul_ps (t1, t2);

                        ans = _mm_add_ps (ans, t1);

                        // printf("%d,%d ",j,class_p->num_each_layer[0]);
                        //  class_p->layers[0]->output_nodes[i] += class_p->layers[0]->weights[i][j] * p->sampleMat[sample_index][j];
                        // printf("%d,%d ",i,sample_index);
                    }
                    // 4个局部和相加
                    ans = _mm_hadd_ps (ans, ans);
                    ans = _mm_hadd_ps (ans, ans);
                    //标量存储
                    float z;
                    _mm_store_ss (&z, ans);
                    class_p->layers[0]->output_nodes[i] += z;
                    class_p->layers[0]->output_nodes[i] += class_p->layers[0]->bias[i];
                    class_p->layers[0]->output_nodes[i] = class_p->layers[0]->activation_function (class_p->layers[0]->output_nodes[i]);
                }
                // printf("@");
                for (int layers_i = 1; layers_i <= class_p->num_layers; layers_i++)
                {

                    for (int i = 0; i < class_p->num_each_layer[layers_i + 1]; i++)
                    {
                        class_p->layers[layers_i]->output_nodes[i] = 0.0;
                        int max_j = class_p->num_each_layer[layers_i];
                        __m128 ans = _mm_setzero_ps();
                        for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 4)
                            // for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                        {
                            __m128 t1, t2;
                            //将内部循环改为4位的向量运算
                            t1 = _mm_loadu_ps (class_p->layers[layers_i]->weights[i] + j);
                            t2 = _mm_loadu_ps (p->sampleMat[sample_index] + j);
                            t1 = _mm_mul_ps (t1, t2);

                            ans = _mm_add_ps (ans, t1);
                        }
                        // 4个局部和相加
                        ans = _mm_hadd_ps (ans, ans);
                        ans = _mm_hadd_ps (ans, ans);
                        //标量存储
                        float z;
                        _mm_store_ss (&z, ans);
                        class_p->layers[layers_i]->output_nodes[i] += z;

                        class_p->layers[layers_i]->output_nodes[i] += class_p->layers[layers_i]->bias[i];
                        class_p->layers[layers_i]->output_nodes[i] = class_p->layers[layers_i]->activation_function (class_p->layers[layers_i]->output_nodes[i]);
                    }
                }
                // printf ("%d finish fw\n", t_id);
                sem_post (&sem_main_after_fw);
                //printf ("%d post fw\n", t_id);
                sample_index++;
            }
            //printf ("%d wait bp with index %d\n", t_id, sample_index);
            sem_wait (&sem_before_bp[t_id]); // 阻塞，等待主线程（操作自己专属的信号量）
            //printf ("%d begin bp in %d with index %d\n", t_id,epoch,sample_index);
            //计算每层的delta
            for (int i = class_p->num_layers - 1; i >= 0; i--)
            {
                float *error = new float[class_p->num_each_layer[i + 1]];
                for (int j = t_id; j < class_p->num_each_layer[i + 1]; j += NUM_THREADS) error[j] = 0.0;
                for (int k = 0; k < class_p->num_each_layer[i + 2]; k++)
                {
                    //for (int j = t_id; j < class_p->num_each_layer[i + 1]; j += NUM_THREADS)
                    int max_j = class_p->num_each_layer[i + 1];
                    __m128 t3 = _mm_set1_ps (class_p->layers[i + 1]->delta[k]);
                    //__m128 ans = _mm_setzero_ps();
                    for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 4)
                    {
                        __m128 t1, t2;
                        t1 = _mm_loadu_ps (error + j);
                        t2 = _mm_loadu_ps (class_p->layers[i + 1]->weights[k] + j);

                        //product=t1*t2*t3，向量对位相乘
                        t2 = _mm_mul_ps (t3, t2);
                        t1 = _mm_add_ps (t1, t2);
                        _mm_storeu_ps (error + j, t1);

                        //    error[j] += class_p->layers[i + 1]->weights[k][j] * class_p->layers[i + 1]->delta[k];
                    }
                }
                int max_j = class_p->num_each_layer[i + 1];
                for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 1)
                    //for (int j = t_id; j < class_p->num_each_layer[i + 1]; j += NUM_THREADS)
                {
                    class_p->layers[i]->delta[j] = error[j] * class_p->layers[i]->derivative_activation_function (class_p->layers[i]->output_nodes[j]);
                }
                delete[]error;
            }
//printf("!");
            //反向传播，weights和bias更新
            for (int k = 0; k < class_p->num_each_layer[1]; k++)
            {
                int max_j = class_p->num_each_layer[0];
                __m128 sr = _mm_set1_ps (class_p->study_rate);
                __m128 t2 = _mm_set1_ps (class_p->layers[0]->delta[k]);
                for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 4)

                    //for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                {
                    __m128 t1, t3, product;
                    t1 = _mm_loadu_ps (class_p->layers[0]->weights[k] + j);
                    t3 = _mm_loadu_ps (p->sampleMat[sample_index - 1] + j);
                    //product=sr*t2*t3，向量对位相乘
                    product = _mm_mul_ps (t3, t2);
                    product = _mm_mul_ps (product, sr);
                    t1 = _mm_sub_ps (t1, product);
                    _mm_storeu_ps (class_p->layers[0]->weights[k] + j, t1);

                    //class_p->layers[0]->weights[k][j] -= class_p->study_rate * p->sampleMat[sample_index - 1][j] * class_p->layers[0]->delta[k];
                }
                class_p->layers[0]->bias[k] -=  class_p->study_rate *  class_p->layers[0]->delta[k];
            }
            //printf("%d here bp with %d/%d\n", t_id, sample_index,NUM_SAMPLE);
            for (int i = 1; i <= class_p->num_layers; i++)
            {
                for (int k = 0; k < class_p->num_each_layer[i + 1]; k++)
                {
                    int max_j = class_p->num_each_layer[i];
                    __m128 sr = _mm_set1_ps (class_p->study_rate);
                    __m128 t2 = _mm_set1_ps (class_p->layers[i]->delta[k]);//printf("1");
                    for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 4)
                        //for (int j = t_id; j < class_p->num_each_layer[i]; j += NUM_THREADS)
                    {
                        __m128 t1, t3, product;
                        t1 = _mm_loadu_ps (class_p->layers[i]->weights[k] + j);
                        t3 = _mm_loadu_ps (class_p->layers[i - 1]->output_nodes + j);
                        //product=sr*t2*t3，向量对位相乘
                        product = _mm_mul_ps (t3, t2);
                        product = _mm_mul_ps (product, sr);
                        t1 = _mm_sub_ps (t1, product);

                        _mm_storeu_ps (class_p->layers[i]->weights[k] + j, t1);

                        //class_p->layers[i]->weights[k][j] -= class_p->study_rate * class_p->layers[i - 1]->output_nodes[j] * class_p->layers[i]->delta[k];
                    }//printf("2");
                    class_p->layers[i]->bias[k] -=  class_p->study_rate * class_p->layers[i]->delta[k];
                }
            }
            // printf ("%d finish bp with %d/%d\n", t_id, sample_index,NUM_SAMPLE);
            sem_post (&sem_main_after_bp);

        }
    }
    pthread_exit (NULL);
}

void* ANN_new::threadFunc_sem (void *param)
{
    threadParam_t *p = (threadParam_t*) param;
    int t_id = p -> t_id;
    ANN_new * class_p = p->class_pointer;
    //printf ("%d begin\n", t_id);
    for (int epoch = 0; epoch < class_p->num_epoch; epoch++)
    {
        int sample_index = 0;
        while (sample_index < NUM_SAMPLE)
        {
            //printf ("%d batch_size=%d,%d\n", t_id, &class_p->batch_size, class_p->batch_size);
            for (int batch_i = 0; batch_i < class_p->batch_size; batch_i++)
            {
                //printf ("%d batch_i=%d,batch_size=%d\n", t_id, batch_i, class_p->batch_size);
                if (sample_index >= NUM_SAMPLE)
                {
                    //printf ("&d,break\n", NUM_SAMPLE);
                    break;
                }
                //printf ("%d wait fw\n", t_id);
                sem_wait (& (sem_before_fw[t_id]) ); // 阻塞,等待主线程（操作自己专属的信号量）
                //printf ("%d begin fw\n", t_id);
                for (int i = 0; i < class_p->num_each_layer[1]; i++)
                {
                    class_p->layers[0]->output_nodes[i] = 0.0;
                    //printf("%d,%d ",i,class_p->num_each_layer[1]);
                    int max_j = class_p->num_each_layer[0];
                    for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 1)
                        // for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                    {
                        // printf("%d,%d ",j,class_p->num_each_layer[0]);
                        class_p->layers[0]->output_nodes[i] += class_p->layers[0]->weights[i][j] * p->sampleMat[sample_index][j];
                        // printf("%d,%d ",i,sample_index);
                    }
                    class_p->layers[0]->output_nodes[i] += class_p->layers[0]->bias[i];
                    class_p->layers[0]->output_nodes[i] = class_p->layers[0]->activation_function (class_p->layers[0]->output_nodes[i]);
                }
                // printf("@");
                for (int layers_i = 1; layers_i <= class_p->num_layers; layers_i++)
                {

                    for (int i = 0; i < class_p->num_each_layer[layers_i + 1]; i++)
                    {
                        class_p->layers[layers_i]->output_nodes[i] = 0.0;
                        int max_j = class_p->num_each_layer[layers_i];
                        for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 1)
                            //for (int j = t_id; j < class_p->num_each_layer[layers_i]; j += NUM_THREADS)
                        {
                            class_p->layers[layers_i]->output_nodes[i] += class_p->layers[layers_i]->weights[i][j] * class_p->layers[layers_i - 1]->output_nodes[j];
                        }
                        class_p->layers[layers_i]->output_nodes[i] += class_p->layers[layers_i]->bias[i];
                        class_p->layers[layers_i]->output_nodes[i] = class_p->layers[layers_i]->activation_function (class_p->layers[layers_i]->output_nodes[i]);
                    }
                }
                //printf ("%d finish fw\n", t_id);
                sem_post (&sem_main_after_fw);
                //printf ("%d post fw\n", t_id);
                sample_index++;
            }
            //printf ("%d wait bp with index %d\n", t_id, sample_index);
            sem_wait (&sem_before_bp[t_id]); // 阻塞，等待主线程（操作自己专属的信号量）
            //printf ("%d begin bp in %d with index %d\n", t_id,epoch,sample_index);
            //计算每层的delta
            for (int i = class_p->num_layers - 1; i >= 0; i--)
            {
                float *error = new float[class_p->num_each_layer[i + 1]];
                for (int j = t_id; j < class_p->num_each_layer[i + 1]; j += NUM_THREADS) error[j] = 0.0;
                for (int k = 0; k < class_p->num_each_layer[i + 2]; k++)
                {
                    //for (int j = t_id; j < class_p->num_each_layer[i + 1]; j += NUM_THREADS)
                    int max_j = class_p->num_each_layer[i + 1];
                    for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 1)
                    {
                        error[j] += class_p->layers[i + 1]->weights[k][j] * class_p->layers[i + 1]->delta[k];
                    }
                }
                int max_j = class_p->num_each_layer[i + 1];
                for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 1)
                    //for (int j = t_id; j < class_p->num_each_layer[i + 1]; j += NUM_THREADS)
                {
                    class_p->layers[i]->delta[j] = error[j] * class_p->layers[i]->derivative_activation_function (class_p->layers[i]->output_nodes[j]);
                }
                delete[]error;
            }

            //反向传播，weights和bias更新
            for (int k = 0; k < class_p->num_each_layer[1]; k++)
            {
                int max_j = class_p->num_each_layer[0];
                for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 1)

                    //for (int j = t_id; j < class_p->num_each_layer[0]; j += NUM_THREADS)
                {
                    class_p->layers[0]->weights[k][j] -= class_p->study_rate * p->sampleMat[sample_index - 1][j] * class_p->layers[0]->delta[k];
                }
                class_p->layers[0]->bias[k] -=  class_p->study_rate *  class_p->layers[0]->delta[k];
            }
            for (int i = 1; i <= class_p->num_layers; i++)
            {
                for (int k = 0; k < class_p->num_each_layer[i + 1]; k++)
                {
                    int max_j = class_p->num_each_layer[i];
                    for (int j = t_id * max_j / NUM_THREADS; j < min (max_j, (t_id + 1) *max_j / NUM_THREADS); j += 1)
                        //for (int j = t_id; j < class_p->num_each_layer[i]; j += NUM_THREADS)
                    {
                        class_p->layers[i]->weights[k][j] -= class_p->study_rate * class_p->layers[i - 1]->output_nodes[j] * class_p->layers[i]->delta[k];
                    }
                    class_p->layers[i]->bias[k] -=  class_p->study_rate * class_p->layers[i]->delta[k];
                }
            }
            //printf ("%d finish bp with %d/%d\n", t_id, sample_index,num_sample);
            sem_post (&sem_main_after_bp);

        }
    }
    pthread_exit (NULL);
}
void ANN_new::train (const int _num_sample, float** _trainMat, float** _labelMat)
{
    creat_params();
    for (int i = 0; i < NUM_THREADS; i++)
    {
        params[i].class_pointer = this;
        for (int j = 0; j < _num_sample; j++)
        {
            for (int k = 0; k < num_each_layer[0]; k++)
                params[i].sampleMat[j][k] = _trainMat[j][k];
        }
    }
    //信号量
    // sem_before_bp = new sem_t[NUM_THREADS]; // 每个线程有自己专属的信号量
    //sem_before_fw = new sem_t[NUM_THREADS];

    int res = 0;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        res += sem_init (& (sem_before_fw[i]), 0, 0);
        res += sem_init (& (sem_before_bp[i]), 0, 0);
    }
    res += sem_init (&sem_main_after_fw, 0, 0);
    res += sem_init (&sem_main_after_bp, 0, 0);
    if (res != 0) printf ("init sem failed!\n");

    //num_sample = _num_sample;
    //创建线程,静态
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        params[t_id].t_id = t_id;
        pthread_create (&handles[t_id], NULL, &test, (void*) & (params[t_id]) );
    }


    //printf ("begin training\n");
    float thre = 1e-2;

    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        if (epoch % 50 == 0)
            printf ("round%d:\n", epoch);
        int sample_index = 0;
        while (sample_index < _num_sample)
        {
            for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
            {
                layers[num_layers]->delta[j] = 0.0;
            }

            for (int batch_i = 0; batch_i < batch_size; batch_i++) //默认batch_size=1，即采用随机梯度下降法，每次使用全部样本训练并更新参数
            {
                if (sample_index >= _num_sample) break;
                //前向传播
                for (int t_i = 0; t_i < NUM_THREADS; t_i++)
                {
                    sem_post (& (sem_before_fw[t_i]) );
                    //printf ("post fw%d sem is %d\n", t_i, sem_before_fw[t_i]);
                }
                //printf ("main is waiting for fw\n");
                for (int t_i = 0; t_i < NUM_THREADS; t_i++)
                {
                    sem_wait (&sem_main_after_fw);
                }
                //predict (_trainMat[index]);
                // printf ("finish predict\n");

                //计算loss，即最后一层的delta,即该minibatch中所有loss的平均值


                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    //均方误差损失函数
                    layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[sample_index][j]) * layers[num_layers]->derivative_activation_function (layers[num_layers]->output_nodes[j]);
                    //交叉熵损失函数
                    //layers[num_layers]->delta[j] += (layers[num_layers]->output_nodes[j] - _labelMat[sample_index][j]);
                }
                // printf ("finish cal error\n");
                sample_index++;
            }

            if (sample_index % batch_size == 0)
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    layers[num_layers]->delta[j] /= batch_size;
                }

            }
            else
            {
                for (int j = 0; j < num_each_layer[num_layers + 1]; j++)
                {
                    layers[num_layers]->delta[j] /= sample_index % batch_size;
                }
            }
            //printf ("main is post for bp\n");
            //反向传播更新参数
            for (int t_i = 0; t_i < NUM_THREADS; t_i++)
            {
                sem_post (&sem_before_bp[t_i]);
            }
            // printf ("main is waiting for bp with index %d\n",sample_index);
            for (int t_i = 0; t_i < NUM_THREADS; t_i++)
            {
                sem_wait (&sem_main_after_bp);
            }

            //back_propagation (_trainMat[index - 1], _labelMat[index - 1]);
            //printf ("finish bp with index:%d\n", sample_index);
        }

        //printf ("finish epoch%d\n", epoch);

    }
    printf ("finish training\n");
    //if(!isNotConver_(_sampleNum, _trainMat，_labelMat, thre)) break;
    // display();
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        pthread_join (handles[t_id], NULL);
    }
    printf ("finish pthread_join\n");
   // delet_params();

   // printf ("finish delet_params\n");
}
bool ANN_new::isNotConver_ (const int _sampleNum, float** _trainMat, float** _labelMat, float _thresh)
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

void ANN_new::predict (float * in)
{
    layers[0]->_forward (in);
    for (int i = 1; i <= num_layers; i++)
    {
        layers[i]->_forward (layers[i - 1]->output_nodes);
    }
}
void ANN_new::display()
{
    for (int i = 0; i <= num_layers; i++)
    {
        layers[i]->display();
    }
}

void ANN_new::back_propagation (float * X, float * Y)
{
    //计算每层的delta
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
        layers[0]->bias[k] -=  study_rate *  layers[0]->delta[k];
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
void ANN_new::get_predictions (float * X)
{
    //for (int i = 0; i < min(5,num_each_layer[0]); i++) printf ("%f,", X[i]);
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
