#include<iostream>
#include<windows.h>
#include <cstdlib>
#include <ctime>

using namespace std;

const int N = 1024*1024;
double *a;
double ans1, ans2, ans3,ans4,ans5;

void recursion(int n){
    if (n == 0)
        return;
    if(n%2)
        for (int i = 0; i <=n / 2; i++)
            a[i] += a[n / 2+ i +1];

    else
        for (int i = 0; i < n / 2; i++)
            a[i] += a[n / 2+ i+1];
    recursion(n/2);

}

void init()			// generate a N*N matrix
{
    a = new double[N];
    for (int i = 0; i < N; i++) {
        a[i] = i;
    }
}
void delet() {
    delete[] a;
}

int main()
{
    long long head, tail, freq;// timers

    init();
    double t1 = 0, t2 = 0, t3 = 0, t4 = 0,t5=0;
    int rep = 10;
    for(int tr = 0; tr < rep; tr++)
    {
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
        ans1 = 0.0;
        init();
        for (int i = 0; i < N; i++) {
            ans1 += a[i];
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
        //cout << (tail - head) * 1000.0 / freq << "ms" << endl;
        t1 += tail - head;
        //cout << endl;

        QueryPerformanceCounter((LARGE_INTEGER*)&head);	// start time
init();
        double sum1=0.0, sum2=0.0,sum3=0.0, sum4=0.0;
        for (int i = 0; i < N; i += 4) {
            sum1 += a[i];
            sum2 += a[i + 1];
            sum3 += a[i+2];
            sum4 += a[i + 3];
        }
        ans2 = sum1 + sum2+sum3 + sum4;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
        //cout << (tail - head) * 1000.0 / freq << "ms" << endl;
        t5 += tail - head;
init();
        sum1=0.0,sum2=0.0;
        for (int i = 0; i < N; i += 2) {
            sum1 += a[i];
            sum2 += a[i + 1];
        }
        ans3 = sum1 + sum2;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
        //cout << (tail - head) * 1000.0 / freq << "ms" << endl;
        t2 += tail - head;
init();
        QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
        for (int m = N; m > 0; m /= 2) // log(n)¸ö²½Öè
            if(m%2)
                for (int i = 0; i <=m / 2; i++)
                    a[i] += a[m / 2+ i +1];
            else
                for (int i = 0; i < m / 2; i++)
                    a[i] += a[m / 2+ i+1];
        ans4 = a[0];
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
       //cout << (tail - head) * 1000.0 / freq << "ms" << endl;
        t3 += tail - head;
        init();
        QueryPerformanceCounter((LARGE_INTEGER*)&head);// start time
        recursion(N);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
       //cout << (tail - head) * 1000.0 / freq << "ms" << endl;
       ans5=a[0];
        t4 += tail - head;
    }
    cout << t1 * 1000.0/ freq / rep <<"ms"<< endl;
    cout << t2 * 1000.0 / freq / rep <<"ms"<< endl;
    cout << t5 * 1000.0 / freq / rep <<"ms"<< endl;
    cout << t3 * 1000.0 / freq / rep << "ms" << endl;
    cout << t4 * 1000.0 / freq / rep << "ms" << endl;
    cout<<ans1<<' '<<ans2<<' '<<ans3<<' '<<ans4<<' '<<ans5<<endl;
    delet();
    return 0;
}
