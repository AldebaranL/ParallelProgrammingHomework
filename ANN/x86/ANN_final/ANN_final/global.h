#ifndef GLOBAL_H
#define GLOBAL_H

extern const int NUM_THREADS;
enum activationType { _sigmoid, _tanh };

extern int NUM_EACH_LAYER[100];
extern int NUM_LAYERS;

extern int NUM_SAMPLE;   
extern float** TRAIN_MAT;
extern float** LABEL_MAT;
extern int MAX_NUM_LAYER;


#endif
