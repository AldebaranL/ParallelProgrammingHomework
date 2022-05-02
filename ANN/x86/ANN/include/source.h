 #ifndef SOURCE_H
#define SOURCE_H

    sem_t *sem_before_bp;// 每个线程有自己专属的信号量
    sem_t *sem_before_fw;
    sem_t sem_main_after_bp;
    sem_t sem_main_after_fw;
#endif
