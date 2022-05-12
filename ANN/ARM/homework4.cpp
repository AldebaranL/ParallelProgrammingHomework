#include <iostream>
#include "ANN_3.h"
#include "ANN_openMP.h"
#include <unistd.h>
#include <sys/time.h>
#include <cstdlib>
#include <ctime>
#include"global.h"
using namespace std;



void creat_samples()
{
    TRAIN_MAT = new float* [NUM_SAMPLE];                         //生成训练样本
    for (int k = 0; k < trainClass; ++k)
    {
        for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i)
        {
            TRAIN_MAT[i] = new float[NUM_EACH_LAYER[0]];
           // srand ( (int) time (0) );
            //TRAIN_MAT[i][0] = ( (rand() % 2000) - 1000) / 10000.0 + k / trainClass;
            for (int j = 0; j < NUM_EACH_LAYER[0]; ++j)
            {

				TRAIN_MAT[i][j] = rand() % 1000 / 10000.0 + 0.1 * (2 * k + 1);
            }
        }
    }//printf("here");
    LABEL_MAT = new float* [NUM_SAMPLE]; //生成标签矩阵
    for (int k = 0; k < trainClass; ++k)
    {
        for (int i = k * numPerClass; i < (k + 1) * numPerClass; ++i)
        {
            LABEL_MAT[i] = new float[NUM_EACH_LAYER[NUM_LAYERS+1]];
            for (int j = 0; j < NUM_EACH_LAYER[NUM_LAYERS+1]; ++j)
            {
               // LABEL_MAT[i][j] = k / trainClass*10+( (rand() % 2000) - 1000) / 10000.0;
                if (j%trainClass == k)
                    LABEL_MAT[i][j] = 1;
                else
                    LABEL_MAT[i][j] = 0;
                //LABEL_MAT[i][j] = ( (rand() % 2000) - 1000 ) / 1000.0;
            }

        }
    }
    printf ("finished creating samples\n");
}
void delet()
{
    printf ("begin delete samples ");
    //释放内存
    for (int i = 0; i < NUM_SAMPLE; ++i)
        delete[] TRAIN_MAT[i];
    delete[] TRAIN_MAT;
    for (int i = 0; i < NUM_SAMPLE; ++i)
        delete[] LABEL_MAT[i];
    delete[] LABEL_MAT;
    printf ("finish delete\n");
}


int main()
{
    creat_samples();
	struct timespec sts,ets;
	time_t dsec;
	long dnsec;

    ANN_3 ann ( (int*) NUM_EACH_LAYER, 128,1,NUM_LAYERS);
    ann.shuffle (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
	timespec_get(&sts, TIME_UTC);
    ann.train(NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
    timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("ori:%ld.%09lds\n",dsec,dnsec);

    ANN_openMP ann2 ( (int*) NUM_EACH_LAYER, 128,1,NUM_LAYERS);
    timespec_get(&sts, TIME_UTC);
    ann2.train (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
    timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("openMP:%ld.%09lds\n",dsec,dnsec);

    float *test_case = new float[10];

    test_case[0]=0.0;
    test_case[1]=0.0;
    //for (int i = 0; i < 2; i++) printf ("%f,", test_case[i]);
    ann.get_predictions(test_case);
    ann2.get_predictions(test_case);
    test_case[0]=0.0;
    test_case[1]=1.0;
    //for (int i = 0; i < 2; i++) printf ("%f,", test_case[i]);
    ann.get_predictions(test_case);
    ann2.get_predictions(test_case);
    test_case[0]=1.0;
    test_case[1]=0.0;
    //for (int i = 0; i < 2; i++) printf ("%f,", test_case[i]);
    ann.get_predictions(test_case);
    ann2.get_predictions(test_case);
    test_case[0]=1.0;
    test_case[1]=1.0;
    //for (int i = 0; i < 2; i++) printf ("%f,", test_case[i]);
    ann.get_predictions(test_case);
    ann2.get_predictions(test_case);

    delet();
//printf("!!");

    return 0;
}

