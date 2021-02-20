#include <iostream>
__global__ void add_vector(float *out, float *a, float *b, int n)
{
    for( int i = 0 ; i < n ; ++i )
    {
        out[i] = a[i] + b[i];
    }
}

void main()
{
    const int N = 10000000;
    float *a, *b, *out;
    float *cuda_a;

    a = (float*)malloc(sizeof(float) * N);
    cudaMalloc((void**)&cuda_a, sizeof(float) * N);
    cudaMemcpy(cuda_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);

    add_vector<<<1,1>>>(out, cuda_a, b, N);
    cudaFree(cuda_a);
    free(a);
}