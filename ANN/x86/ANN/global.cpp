
#include "global.h"

const int NUM_THREADS = 4;
sem_t *sem_before_bp = new sem_t[NUM_THREADS]; // 每个线程有自己专属的信号量
sem_t *sem_before_fw = new sem_t[NUM_THREADS];
sem_t sem_main_after_bp;
sem_t sem_main_after_fw;





//int NUM_EACH_LAYER[10] = {4,1024*2, 1024*2,1024*2, 4};
int NUM_EACH_LAYER[10] = {256,256,256};
int NUM_LAYERS=1;

const int trainClass = 4; //类别数
const int numPerClass = 16;  //每个类别的样本点数

const int NUM_SAMPLE = trainClass * numPerClass;     //总的样本数=每类训练样本数*类别数
float** TRAIN_MAT;
float** LABEL_MAT;

pthread_t *handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle

threadParam_t *params = new threadParam_t[NUM_THREADS]; // 创建对应的线程数据结构

pthread_barrier_t *barrier_fw=new pthread_barrier_t[NUM_LAYERS+1];
pthread_barrier_t *barrier_bp=new pthread_barrier_t[NUM_LAYERS+1];
pthread_barrier_t *barrier_delta=new pthread_barrier_t[NUM_LAYERS+1];
pthread_barrier_t barrier_before_bp;

void creat_params()
{
    for (int i = 0; i < NUM_THREADS; i++)
    {
        params[i].sampleMat = new float*[NUM_SAMPLE];
        for (int j = 0; j < NUM_SAMPLE; j++)
        {
            params[i].sampleMat[j] = new float[NUM_EACH_LAYER[0]];
        }

    }
}
void delet_params(){
   // printf("0");
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
             //printf("%d ",t_id);
            for (int j = 0; j < NUM_SAMPLE; j++)
            {
                //printf("%d,",j);
                delete []params[t_id].sampleMat[j];
            }
            // printf("%d",t_id);
            delete[]params[t_id].sampleMat;
        }

        delete []handles;
        delete []params;
//printf("1");
        //销毁所有信号量
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_destroy (&sem_before_fw[i]);
            sem_destroy (&sem_before_bp[i]);
        }
       delete[]sem_before_fw;
        delete[]sem_before_bp;
       //printf("2");
        sem_destroy (&sem_main_after_fw);
        sem_destroy (&sem_main_after_bp);
//printf("3");
}
//sem_t	sem_parent;
//sem_t	sem_children;
