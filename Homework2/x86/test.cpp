/*#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

int main(void)
{
    int* p1 = (int*)malloc(10 * sizeof * p1);
    printf("default-aligned addr:   %p\n", (void*)p1);
    free(p1);

    int* p2 = (int*)_aligned_malloc(1024 * sizeof * p2,1024*1024);
    printf("1024-byte aligned addr: %p\n", (void*)p2);
    _aligned_free(p2);
    return 0;
}
*/
