#include <iostream>
#include"ANN_4.h"
#include"ANN_parallel.h"
#include<windows.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include"global.h"
using namespace std;


void creat_samples()
{
    TRAIN_MAT = new float* [NUM_SAMPLE];                         //生成训练样本
    for (int i = 0; i < NUM_SAMPLE; ++i)
    {
        TRAIN_MAT[i] = new float[NUM_EACH_LAYER[0]];
    }
    LABEL_MAT = new float* [NUM_SAMPLE]; //生成标签矩阵
    for (int i = 0; i < NUM_SAMPLE; ++i)
    {
        LABEL_MAT[i] = new float[NUM_EACH_LAYER[NUM_LAYERS + 1]];
        for (int j = 0; j < NUM_EACH_LAYER[NUM_LAYERS + 1]; ++j)
        {
            LABEL_MAT[i][j] = 0;
        }
    }
    MAX_NUM_LAYER = NUM_EACH_LAYER[0];
    for (int i = 1; i <= NUM_LAYERS + 1; i++) {
        MAX_NUM_LAYER = max(NUM_EACH_LAYER[i], MAX_NUM_LAYER);
    }
    //MAX_NUM_LAYER = 256;
    printf("finished creating samples\n");

}
void print_samples() {
    for (int i = 0; i < min(5, NUM_SAMPLE); i++)
    {
        for (int j = 0; j < min(5, NUM_EACH_LAYER[0]); j++) printf("%f ", TRAIN_MAT[i][j]); printf("\n");
        for (int j = 0; j < min(5, NUM_EACH_LAYER[NUM_LAYERS+1]); j++) printf("%f ", LABEL_MAT[i][j]);
        printf("\n");
    }
}
void delet_samples()
{
    printf("begin delete samples ");
    //释放内存
    for (int i = 0; i < NUM_SAMPLE; ++i)
        delete[] TRAIN_MAT[i];
    delete[] TRAIN_MAT;
    for (int i = 0; i < NUM_SAMPLE; ++i)
        delete[] LABEL_MAT[i];
    delete[] LABEL_MAT;
    printf("finish delete\n");
}

void read_samples1(const char* filename) {
    ifstream inFile(filename, ios::in);
    if (!inFile)
    {
        cout << "打开文件失败！" << endl;
        exit(1);
    }
    NUM_SAMPLE = 0;
    string line;
    getline(inFile, line);
    NUM_EACH_LAYER[0] = 4;
    NUM_EACH_LAYER[NUM_LAYERS + 1] = 3;

    while (getline(inFile, line))//getline(inFile, line)表示按行读取CSV文件中的数据
    {
        string field;
        istringstream sin(line); //将整行字符串line读入到字符串流sin中 
        for (int j = 0; j < NUM_EACH_LAYER[0]; j++) {
            getline(sin, field, ',');
            TRAIN_MAT[NUM_SAMPLE][j] = atof(field.c_str());
        }
        getline(sin, field);
        LABEL_MAT[NUM_SAMPLE][atoi(field.c_str())] = 1;
        NUM_SAMPLE++;
    }
    inFile.close();
}
void read_samples(const char* filename) {
    ifstream inFile(filename, ios::in);
    if (!inFile)
    {
        cout << "打开文件失败！" << endl;
        exit(1);
    }
    NUM_SAMPLE = 0;
    string line;
    NUM_EACH_LAYER[0] = 9;
    NUM_EACH_LAYER[NUM_LAYERS+1] = 2;

    while (getline(inFile, line))//getline(inFile, line)表示按行读取CSV文件中的数据
    {
        string field;
        istringstream sin(line); //将整行字符串line读入到字符串流sin中 
        getline(sin, field, ',');
        for (int j = 0; j < NUM_EACH_LAYER[0]; j++) {
            getline(sin, field, ',');
            TRAIN_MAT[NUM_SAMPLE][j] = atof(field.c_str());
        }
        getline(sin, field);
        LABEL_MAT[NUM_SAMPLE][atoi(field.c_str())/2-1] = 1;
        NUM_SAMPLE++;
    }
    inFile.close();
}
int main(const int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);//MPI_THREAD_MULTIPLE
    int myid=0, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);    


    int thesize[10] = { 32,64,96,128,256, 512,768,1024,1280 };

    for (int d_i = 0; d_i < 4; d_i++)
    {
        for (int l_i = 1; l_i < NUM_LAYERS + 1; l_i++) NUM_EACH_LAYER[l_i] = thesize[d_i];
        
        //if (myid == 0) {
        creat_samples();
            read_samples1("D:/Mycodes/VSProject/ANN_final/ANN_final/iris_training.csv");
            //read_samples("D:/Mycodes/VSProject/ANN_final/ANN_final/breast-cancer-wisconsin.data");
            //print_samples();
        //}
         if (myid == 0) cout << "---------------------begin " << thesize[d_i] << "--------------------" << endl;
        for (int l_i = 0; l_i < NUM_LAYERS + 2; l_i++) cout << NUM_EACH_LAYER[l_i] << ' '; cout << endl;
        //MPI_Barrier(MPI_COMM_WORLD);
       // cout << endl;
        ANN_parallel ann((int*)NUM_EACH_LAYER, 30, 128, NUM_LAYERS, 0.01);
        ann.shuffle(NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
       // ANN_parallel ann2((int*)NUM_EACH_LAYER, 50, 1, NUM_LAYERS, 0.005);
       // ann2.shuffle(NUM_SAMPLE, TRAIN_MAT, LABEL_MAT);
        int train_num = NUM_SAMPLE * 0.8;
        MPI_Barrier(MPI_COMM_WORLD);
        ann.train_MPI(train_num, TRAIN_MAT, LABEL_MAT);
        MPI_Barrier(MPI_COMM_WORLD);
        //ann.train_MPI_openMP_SIMD(train_num, TRAIN_MAT, LABEL_MAT);
        if (myid == 0) {
            ann.train(train_num, TRAIN_MAT, LABEL_MAT);
            //ann.train_SIMD(train_num, TRAIN_MAT, LABEL_MAT);
            //ann.train_openMP_SIMD(train_num, TRAIN_MAT, LABEL_MAT);
            //ann.train_openMP(train_num, TRAIN_MAT, LABEL_MAT);
            //ann.get_results(NUM_SAMPLE - train_num, &TRAIN_MAT[train_num], &LABEL_MAT[train_num]);
            delet_samples();
        }
    }
    MPI_Finalize();
    //printf("!!");
    std::fflush(stdout);


    return 0;
}

