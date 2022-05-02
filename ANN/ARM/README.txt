//arm-linux-gnueabi-g++ -march=armv8-a -mfloat-abi=hard homework2.cpp
aarch64-linux-gnu-g++ -o test -march=armv8.2-a homework2.cpp
qemu-aarch64 -L /usr/aarch64-linux-gnu ./test