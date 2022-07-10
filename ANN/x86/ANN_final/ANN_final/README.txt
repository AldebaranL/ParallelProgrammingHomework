1.编译
2.进入cmd
cd /d D:\Mycodes\VSProject\ANN_final\Debug
start /b smpd -d
mpiexec -hosts 1 localhost 4 .\ANN_final.exe
或者
mpiexec -n .\hello.exe
