///*#include <stdio.h>
//#include <stdlib.h>
//#include <immintrin.h>
//
//int main(void)
//{
//    int* p1 = (int*)malloc(10 * sizeof * p1);
//    printf("default-aligned addr:   %p\n", (void*)p1);
//    free(p1);
//
//    int* p2 = (int*)_aligned_malloc(1024 * sizeof * p2,1024*1024);
//    printf("1024-byte aligned addr: %p\n", (void*)p2);
//    _aligned_free(p2);
//    return 0;
//}
//*/
//#include <stdio.h>
//#include <stdlib.h>
//#include <pthread.h>
//#include <semaphore.h>
//#include"global.h"
//#define	NUM_THREADS	4
//
//
//static void *test (void * arg);
//class Test_class
//{
//public:
//    Test_class(){
//        ;
//    }
//    typedef struct
//    {
//        int	threadId;
//    } threadParm_t;
//    ;
//    ~Test_class(){
//        ;
//    }
//
//    void *threadFunc (void *parm)
//    {
//        threadParm_t	*p = (threadParm_t *) parm;
//
//        fprintf (stdout, "I am the child thread %d.\n", p->threadId);
//        sem_post (&sem_parent);
//        sem_wait (&sem_children);
//        fprintf (stdout, "Thread %d is going to exit.\n", p->threadId);
//        pthread_exit (NULL);
//    }
//    void test_func()
//    {
//        sem_init (&sem_parent, 0, 0);
//        sem_init (&sem_children, 0, 0);
//        pthread_t	thread[NUM_THREADS];
//        threadParm_t	threadParm[NUM_THREADS];
//        int	i;
//        for (i = 0; i < NUM_THREADS; i++)
//        {
//            threadParm[i].threadId = i;
//            pthread_create (&thread[i], NULL, &test, (void *) &threadParm[i]);
//        }
//
//        for (i = 0; i < NUM_THREADS; i++)
//        {
//            sem_wait (&sem_parent);
//        }
//
//        fprintf (stdout, "All the child threads has printed.\n");
//
//        for (i = 0; i < NUM_THREADS; i++)
//        {
//            sem_post (&sem_children);
//        }
//        for (i = 0; i < NUM_THREADS; i++)
//        {
//            pthread_join (thread[i], NULL);
//        }
//
//        sem_destroy (&sem_parent);
//        sem_destroy (&sem_children);
//    }
//};
//static void *test (void * arg)
//{
//    return static_cast<Test_class *> (arg)->threadFunc (arg);
//}
//
//int main (int argc, char *argv[])
//{
//    Test_class *test_class=new Test_class();
//    test_class->test_func();
//
//    return 0;
//}
