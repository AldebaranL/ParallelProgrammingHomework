#include <iostream>
#include "ANN_2.h"
#include "ANN_pthread.h"
//#include"ANN_SIMD.h"
//#include"ANN_SIMD_aligned.h"
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

                //TRAIN_MAT[i][j] = ( (rand() % 2000) - 1000 ) / 1000.0;
                //printf("%d",k);
                /*
                switch(k){
                case 0:
                    TRAIN_MAT[i][j] = 0.0;
                    break;
                case 1:
                    TRAIN_MAT[i][j] = (float)j;
                    break;
                case 2:
                    TRAIN_MAT[i][j] = 1.0-(float)j;
                    break;
                case 3:
                    TRAIN_MAT[i][j] = 1.0;
                    break;
                default:
                    printf("?");
                }//*/
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
                /*
                switch(k){
                case 0:
                LABEL_MAT[i][j] = 1.0;
                break;
                case 1:
                LABEL_MAT[i][j] = 1.0;
                break;
                case 2:
                LABEL_MAT[i][j] = 0.0;
                break;
                case 3:
                LABEL_MAT[i][j] = 1.0;
                break;
                default:
                printf("?");
                }
                //*/
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
//    for (int i = 0; i <min(5, NUM_SAMPLE); i++)
//    {
//        for(int j=0;j<min(5,NUM_EACH_LAYER[0]);j++)printf("%f ",TRAIN_MAT[i][j]);printf("\n");
//        for (int j = 0; j < min(5,NUM_EACH_LAYER[2]); j++)
//        {
//            printf("%f ",LABEL_MAT[i][j]);
//        }
//         printf("\n");
//    }
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

    ANN_2 ann2 ( (int*) NUM_EACH_LAYER, 128,1,NUM_LAYERS);
    ann2.shuffle (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);

//        for (int i = 0; i <min(5, NUM_SAMPLE); i++)
//    {
//        for(int j=0;j<min(5,NUM_EACH_LAYER[0]);j++)printf("%f ",TRAIN_MAT[i][j]);printf("\n");
//        for (int j = 0; j < min(5,NUM_EACH_LAYER[2]); j++)
//        {
//            printf("%f ",LABEL_MAT[i][j]);
//        }
//         printf("\n");
//    }
	struct timespec sts,ets;
	time_t dsec;
	long dnsec;

	timespec_get(&sts, TIME_UTC);
    ann2.train_SIMD (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
    timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("SIMD:%ld.%09lds\n",dsec,dnsec);

    ANN_2 ann3 ( (int*) NUM_EACH_LAYER, 128,1,NUM_LAYERS);
    timespec_get(&sts, TIME_UTC);
    ann3.train (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
    timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("ori:%ld.%09lds\n",dsec,dnsec);

    ANN_pthread ann ( (int*) NUM_EACH_LAYER,  128,1,NUM_LAYERS);
    timespec_get(&sts, TIME_UTC);
    ann.train_sem (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
        timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("pthreadsem:%ld.%09lds\n",dsec,dnsec);


    ANN_pthread ann4 ( (int*) NUM_EACH_LAYER,  128,1,NUM_LAYERS);
    timespec_get(&sts, TIME_UTC);
    ann4.train_semSIMD (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
        timespec_get(&ets, TIME_UTC);
    dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
		dsec--;
		dnsec+=1000000000ll;
	}
	printf ("pthreadsemSIMD:%ld.%09lds\n",dsec,dnsec);

    float *test_case = new float[10];
 /*
  srand ( (int) time (0) );
 test_case[0] = ( (rand() % 2000) - 1000) / 10000.0 + 0 / trainClass;
    for (int j = 1; j < NUM_EACH_LAYER[0]; ++j)
    {
        test_case[j] = ( (rand() % 2000) - 1000 ) / 1000.0;
    }
    //ann.get_predictions (test_case);
    ann2.get_predictions (test_case);

    test_case[0] = ( (rand() % 2000) - 1000) / 10000.0 + 2 / trainClass;
    for (int j = 1; j < NUM_EACH_LAYER[0]; ++j)
    {
        test_case[j] = ( (rand() % 2000) - 1000 ) / 1000.0;
    }
    ann.get_predictions (test_case);
    ann2.get_predictions (test_case);
    */
   // /*
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
  //  */
//
//    long long head, tail, freq;// timers
//    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
//
//
//    ANN_SIMD_aligned ann_simd_classify2(sampleN, inNodes, outNodes, hidnodes, 0.12);
//    printf("here");
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
//	ann_simd_classify2.train_avx(sampleN, trainMat, labelMat);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
//    cout <<"aligned avx:"<< (tail - head) * 1000.0 / freq << "ms" << endl;
//
//
//
//    ANN_SIMD_aligned ann_classify(sampleN, inNodes, outNodes, hidnodes, 0.12);
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
//	ann_classify.train_sse(sampleN, trainMat, labelMat);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
//    cout <<"aligned sse:"<< (tail - head) * 1000.0 / freq << "ms" << endl;
//
//    ANN_SIMD ann_simd_classify3(sampleN, inNodes, outNodes, hidnodes, 0.12);
//
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
//	ann_simd_classify3.train_avx(sampleN, trainMat, labelMat);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
//    cout <<"avx:"<< (tail - head) * 1000.0 / freq << "ms" << endl;
//    //
//	ANN_SIMD ann_simd_classify1(sampleN, inNodes, outNodes, hidnodes, 0.12);  //输入层为inNodes个结点，输出层outNodes个结点，单层隐藏层,studyRate为0.12
//
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
//	ann_simd_classify1.train_sse(sampleN, trainMat, labelMat);
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
//    cout <<"sse:"<< (tail - head) * 1000.0 / freq << "ms" << endl;
//
//
//
//	for (int i = 0; i < 30; ++i) {
//		ann_classify.predict(trainMat[i + 120], NULL);
//		//std::cout << std::endl;
//	}

    delet();
//printf("!!");

    return 0;
}

