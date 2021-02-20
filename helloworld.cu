#include <stdio.h>

__global__ void hello()
{
    printf("Hello World from GPU!");
}

int main()
{
    hello<<<1,1>>>();
    return 0;
}