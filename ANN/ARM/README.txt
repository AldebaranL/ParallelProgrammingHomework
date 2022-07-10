homework2(SMID):
//arm-linux-gnueabi-g++ -march=armv8-a -mfloat-abi=hard homework2.cpp
aarch64-linux-gnu-g++ -o test -march=armv8.2-a homework2.cpp
qemu-aarch64 -L /usr/aarch64-linux-gnu ./test

homework3(pthread):
aarch64-linux-gnu-g++ -pthread -o test -march=armv8.2-a Layer.h Layer.cpp ANN_2.h ANN_pthread.h global.h global.cpp ANN_2.cpp ANN_pthread.cpp homework3.cpp
qemu-aarch64 -L /usr/aarch64-linux-gnu ./test

homework4(openMP):
aarch64-linux-gnu-g++ -pthread -o test -march=armv8.2-a Layer.h Layer.cpp ANN_3.h ANN_openMP.h global.h global.cpp ANN_3.cpp ANN_openMP.cpp homework4.cpp
qemu-aarch64 -L /usr/aarch64-linux-gnu ./test
//20220615注：原global.cpp与global.h现已改名为_global.cpp与_global.h！！！

homework5(MPI)
mpic++ Layer.h Layer.cpp ANN_MPI.h global.h global.cpp ANN_MPI.cpp homework5.cpp
mpiexec -n 2 ./a.out
mpiexec -hosts 1 localhost 4 ./a.out

windows下：
cd /d D:\Mycodes\VSProject\ANN_MPI\Debug
start /b smpd -d
mpiexec -hosts 1 localhost 4 .\ANN_MPI.exe

homework5(MPI)
windows下：
cd /d D:\Mycodes\VSProject\ANN_final\Debug
start /b smpd -d
mpiexec -hosts 1 localhost 4 .\ANN_final.exe