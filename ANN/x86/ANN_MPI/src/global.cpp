#include "global.h"

int NUM_THREADS = 4;

//int NUM_EACH_LAYER[10] = {4,1024*2, 1024*2,1024*2, 4};
int NUM_EACH_LAYER[10] = { 1,1,1,1,1 };
int NUM_LAYERS = 1;

const int trainClass = 4; //类别数
const int numPerClass = 16;  //每个类别的样本点数

const int NUM_SAMPLE = trainClass * numPerClass;     //总的样本数=每类训练样本数*类别数
float** TRAIN_MAT;
float** LABEL_MAT;

