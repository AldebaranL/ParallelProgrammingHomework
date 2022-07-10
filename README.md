# Parallel-programming-Homework
南开大学2022春 并行程序设计作业。

Homework1 是cache与流水线优化，接下来的实验均为对ANN的优化。

Homework2 ANN.h中ANN为原始类，仅有一个隐藏层，SIMD实验在此基础上进行优化，实现了SSE，AVX，NOEN，分别有对齐与不对齐两个版本，具见ANN_SIMD.h与ANN_SIMD_aligned。

Homework3 2022/5/2 对ANN类进行改进，ANN2.h为改进后的类，可指定任意多隐藏层;训练时可指定batch_size与epoch_num;使用独立的Layer，具见Layer.h。pthread实验在此基础上完成，也同时实现了单独的sse(noen)的SIMD优化与pthread+sse(noen)优化,更多参见实验报告。

Homework4 ANN_3与ANN_2完全相同，把train的三个循环整合到的一个函数中，方便进行改进。在ANN_3的基础上进行openMP优化。

(Homework2-Homework4:x86下的实验代码均在 ANN\x86\ANN中。)

Homework5 20220617 在ANN_3的基础上进行MPI优化。x86下代码均在 ANN\x86\ANN_MPI中。

期末大作业 20220710 在ANN_4的基础上进行整体优化，结构进行调整，使用鸢尾花和乳腺癌两个数据集进行测试和实验。代码在 ANN\x86\ANN_final中。
