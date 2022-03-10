#include<iostream>
#include <unistd.h>
#include <sys/time.h>
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
    struct timeval head,tail;

    init();
    double t1 = 0, t2 = 0, t3 = 0, t4 = 0,t5 = 0;
    int rep = 10;
    for(int tr = 0; tr < rep; tr++)
    {
        gettimeofday(&head,NULL);
        ans1 = 0.0;
        for (int i = 0; i < N; i++) {
            ans1 += a[i];
        }
        gettimeofday(&tail,NULL);
        t1 += tail.tv_sec*1000.0 - head.tv_sec*1000.0 + (tail.tv_usec - head.tv_usec)/1000.0;
        gettimeofday(&head,NULL);
        double sum1=0.0, sum2=0.0,sum3=0.0, sum4=0.0;
        for (int i = 0; i < N; i += 4) {
            sum1 += a[i];
            sum2 += a[i + 1];
            sum3 += a[i+2];
            sum4 += a[i + 3];
        }
        ans2 = sum1 + sum2+sum3 + sum4;
        gettimeofday(&tail,NULL);
        t5 += tail.tv_sec*1000.0 - head.tv_sec*1000.0 + (tail.tv_usec - head.tv_usec)/1000.0;
        sum1=0.0, sum2=0.0;
        for (int i = 0; i < N; i += 2) {
            sum1 += a[i];
            sum2 += a[i + 1];
        }
        ans3 = sum1 + sum2;
        gettimeofday(&tail,NULL);
        t2 += tail.tv_sec*1000.0 - head.tv_sec*1000.0 + (tail.tv_usec - head.tv_usec)/1000.0;
        gettimeofday(&head,NULL);
        for (int m = N; m > 0; m /= 2) // log(n)¸ö²½Öè
            if(m%2)
                for (int i = 0; i <=m / 2; i++)
                    a[i] += a[m / 2+ i +1];
            else
                for (int i = 0; i < m / 2; i++)
                    a[i] += a[m / 2+ i+1];
        ans4 = a[0];
        gettimeofday(&tail,NULL);
        t3 += tail.tv_sec*1000.0 - head.tv_sec*1000.0 + (tail.tv_usec - head.tv_usec)/1000.0;
        gettimeofday(&head,NULL);
        recursion(N);
		ans5 = a[0];
        gettimeofday(&tail,NULL);
        t4 += tail.tv_sec*1000.0 - head.tv_sec*1000.0 + (tail.tv_usec - head.tv_usec)/1000.0;

    }
    cout << t1  / rep <<"ms"<< endl;
    cout << t2/ rep <<"ms"<< endl;
    cout << t5/ rep <<"ms"<< endl;
    cout << t3 / rep << "ms" << endl;
    cout << t4   / rep << "ms" << endl;
	cout<<ans1<<' '<<ans2<<' '<<ans3<<' '<<ans4<<' '<<ans5<<endl;
    delet();
    return 0;
}
