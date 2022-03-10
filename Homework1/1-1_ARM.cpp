#include<iostream>
#include <unistd.h>
#include <sys/time.h>
#include <cstdlib>
#include <ctime>

using namespace std;

const int N = 1024*4;
double **a, *b;
double *inner_product1,* inner_product2;

void init()			// generate a N*N matrix
{
    a = new double*[N];
    b = new double[N];
    inner_product1 = new double[N];
    inner_product2 = new double[N];
    for (int i = 0; i < N; i++) {
        a[i] = new double[N];
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            a[i][j] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX)+ 10E-100;
        b[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX) + 10E-100;
    }

}
void delet() {
    delete[] b;
    delete[] inner_product1;
    delete[] inner_product2;
    for (int i = 0; i < N; i++) {
        delete[] a[i];
    }
    delete[] a;
}

void check() {
    for (int i = 0; i < N; i++) {
        if (inner_product1[i] != inner_product2[i])
            cout << inner_product1[i] << "，" << inner_product2[i] << endl;
    }
}
int main()
{
    //long long head, tail, freq;// timers
    struct timeval head,tail;
    double timeuse;
    double timeuse_ms;

    init();
    double t1 = 0, t2 = 0;
    int rep = 10;
    for(int tr = 0; tr < rep; tr++)
    {
        gettimeofday(&head,NULL);
        for (int i = 0; i < N; i++) {
            inner_product1[i] = 0.0;
            for (int j = 0; j < N; j++)
                inner_product1[i] = b[j]*a[j][i]+inner_product1[i];
            //cout << inner_product1[i];
        }
        gettimeofday(&tail,NULL);
        //cout << (tail - head) * 1000.0 / freq << "ms" << endl;
        t1 += tail.tv_sec*1000.0 - head.tv_sec*1000.0 + (tail.tv_usec - head.tv_usec)/1000.0;
        //cout << endl;
        gettimeofday(&head,NULL);
        for (int i = 0; i < N; i++)
            inner_product2[i] = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                inner_product2[j] = b[i]*a[i][j]+inner_product2[j];
        }
        //for (int i = 0; i < N; i++) cout << inner_product2[i];
        gettimeofday(&tail,NULL);
        //cout << (tail - head) * 1000.0 / freq << "ms" << endl;
        t2 += tail.tv_sec*1000.0 - head.tv_sec*1000.0 + (tail.tv_usec - head.tv_usec)/1000.0;

        check();
    }
    cout << t1 /rep <<"ms"<< endl;
    cout << t2 /rep <<"ms"<< endl;
    delet();
    return 0;
}
