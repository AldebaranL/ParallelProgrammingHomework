homework2:
//arm-linux-gnueabi-g++ -march=armv8-a -mfloat-abi=hard homework2.cpp
aarch64-linux-gnu-g++ -o test -march=armv8.2-a homework2.cpp
qemu-aarch64 -L /usr/aarch64-linux-gnu ./test

homework3:
aarch64-linux-gnu-g++ -pthread -o test -march=armv8.2-a Layer.h Layer.cpp ANN_2.h ANN_pthread.h global.h global.cpp ANN_2.cpp ANN_pthread.cpp homework3.cpp
qemu-aarch64 -L /usr/aarch64-linux-gnu ./test