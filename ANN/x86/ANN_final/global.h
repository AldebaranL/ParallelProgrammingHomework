#ifndef GLOBAL_H
#define GLOBAL_H

//#include<semaphore.h>
//#include<pthread.h>
//#include"ANN_pthread.h"
//#include <iostream>
//using namespace std;
class ANN_pthread;
typedef struct
{
    int t_id; //线程 id
    float** sampleMat;
    ANN_pthread* class_pointer;
} threadParam_t;

enum activationType { _sigmoid, _tanh };
extern const int NUM_THREADS;

//extern   threadParam_t* params;

extern const int trainClass; //类别数
extern const int numPerClass;  //每个类别的样本点数

extern int NUM_EACH_LAYER[100];
extern int NUM_LAYERS;

extern int NUM_SAMPLE;     //总的样本数=每类训练样本数*类别数
extern float** TRAIN_MAT;
extern float** LABEL_MAT;
extern int MAX_NUM_LAYER;
//void creat_params();
//void delet_params();

#endif
