#ifndef GLOBAL_H
#define GLOBAL_H

#include<semaphore.h>
#include<pthread.h>
#include"ANN_pthread.h"
//#include <iostream>
//using namespace std;
class ANN_pthread;
typedef struct
{
    int t_id; //线程 id
    float **sampleMat;
    ANN_pthread* class_pointer;
} threadParam_t;
extern const int NUM_THREADS;
extern sem_t *sem_before_bp;// 每个线程有自己专属的信号量
extern sem_t *sem_before_fw;
extern sem_t sem_main_after_bp;
extern sem_t sem_main_after_fw;

extern pthread_barrier_t b1;

extern    pthread_t *handles;
extern   threadParam_t *params;

extern const int trainClass ; //类别数
extern const int numPerClass;  //每个类别的样本点数

extern int NUM_EACH_LAYER[10];
extern int NUM_LAYERS;

extern const int NUM_SAMPLE;     //总的样本数=每类训练样本数*类别数
extern float** TRAIN_MAT;

extern pthread_barrier_t *barrier_fw;
extern pthread_barrier_t *barrier_bp;
extern pthread_barrier_t *barrier_delta;
extern pthread_barrier_t barrier_before_bp;

extern float** LABEL_MAT;
//extern  sem_t	sem_parent;
//extern  sem_t	sem_children;
void creat_params();
void delet_params();

#endif
