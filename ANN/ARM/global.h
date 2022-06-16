#ifndef GLOBAL_H
#define GLOBAL_H

extern int NUM_THREADS;


extern const int trainClass; //类别数
extern const int numPerClass;  //每个类别的样本点数

extern int NUM_EACH_LAYER[10];
extern int NUM_LAYERS;

extern const int NUM_SAMPLE;     //总的样本数=每类训练样本数*类别数
extern float** TRAIN_MAT;

extern float** LABEL_MAT;

#endif
