#include <iostream>
#include"ANN_MPI.h"
#include <cstdlib>
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
            //srand ( (int) time (0) );
            srand (i);
            //  TRAIN_MAT[i][0] = ( (rand() % 2000) - 1000) / 10000.0 + k / trainClass;
            for (int j = 0; j < NUM_EACH_LAYER[0]; ++j)
            {
                TRAIN_MAT[i][j] = rand() % 10000 / 10000.0 + (200 * k + 1);

                // TRAIN_MAT[i][j] = ( (rand() % 20000) - 10000 );// / 1000.0;
                // printf("%d",k);
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
            LABEL_MAT[i] = new float[NUM_EACH_LAYER[NUM_LAYERS + 1]];
            for (int j = 0; j < NUM_EACH_LAYER[NUM_LAYERS + 1]; ++j)
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
                */
                // LABEL_MAT[i][j] = k / trainClass*10+( (rand() % 2000) - 1000) ;/// 10000.0;
                if (j % trainClass == k)
                    LABEL_MAT[i][j] = 1;
                else
                    LABEL_MAT[i][j] = 0;
                // LABEL_MAT[i][j] = ( (rand() % 2000) - 1000 );// / 1000.0;
            }

        }
    }
    printf ("finished creating samples\n");
    //    for (int i = 0; i < min (5, NUM_SAMPLE); i++)
    //    {
    //        printf ("sample%d:\ntrainmat:\n",i);
    //        for (int j = 0; j < min (5, NUM_EACH_LAYER[0]); j++) printf ("%f ", TRAIN_MAT[i][j]);
    //        printf ("\nlabelmat:\n");
    //        for (int j = 0; j < min (5, NUM_EACH_LAYER[2]); j++)
    //        {
    //            printf ("%f ", LABEL_MAT[i][j]);
    //        }
    //        printf ("\n\n");
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


int main (int argc, char* argv[])
{
    //MPI_Init(&argc, &argv);

    int provided;
    MPI_Init_thread (&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int myid, numprocs;
    MPI_Comm_rank (MPI_COMM_WORLD, &myid);
    int thesize[10] = { 256, 512,768,1024,1280,2048 };
    for (int d_i = 0; d_i < 5; d_i++)
    {
        for (int l_i = 0; l_i < NUM_LAYERS + 2; l_i++) NUM_EACH_LAYER[l_i] = thesize[d_i];
        if (myid == 0) cout << "---------------------begin " << thesize[d_i] << "--------------------" << endl;

        creat_samples();
for (int l_i = 0; l_i < NUM_LAYERS + 2; l_i++) cout<<NUM_EACH_LAYER[l_i]<<' ';
        ANN_MPI ann ( (int*) NUM_EACH_LAYER, 128, 1, NUM_LAYERS, 0.1);
       // cout<<"here"<<endl;
        ann.shuffle (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
//cout<<"here22"<<endl;
        if(myid==0)ann.train (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
        //ann.get_predictions (TRAIN_MAT[0]);
        MPI_Barrier (MPI_COMM_WORLD);//cout<<"here23"<<endl;
        ann.train_MPI_predict (NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);

        MPI_Barrier (MPI_COMM_WORLD);
        //ann.train_MPI_all_static(NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
        //ann.get_predictions (TRAIN_MAT[0]);

        //cout << "---------------------end " << thesize[d_i] << "--------------------" << endl;
        delet();
    }
    MPI_Finalize();

    //printf("!!");
    fflush (stdout);
    return 0;
}

