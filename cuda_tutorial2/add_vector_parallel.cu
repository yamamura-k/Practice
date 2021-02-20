#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_vector(float *out, float *a, float *b, int n)
{
    int index = threadIdx.x;
    int stride = blockDim.x;

    for( int i = index ; i < n ; i += stride )
    {
        out[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 100000;
    float *a, *b, *out;
    float *cuda_a, *cuda_b, *cuda_out;

    std::cout << "This program computes sum of " << N << " dimensional vectors." << std::endl;
    std::cout << "All variables are written in source code." << std::endl;

    // allocate memory on cpu for vector a, b, out
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    
    // initialization
    for( int i = 0 ; i < N ; i++ )
    {
        a[i] = i*1.1;
        b[i] = N*0.6-i;
        out[i] = 0.0;
    }

    // allocate memory on gpu for vector cuda_a, cuda_b, cuda_out
    cudaMalloc((void**)&cuda_a, sizeof(float) * N);
    cudaMalloc((void**)&cuda_b, sizeof(float) * N);
    cudaMalloc((void**)&cuda_out, sizeof(float) * N);

    // copy memory from cpu to gpu
    cudaMemcpy(cuda_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);

    // main procedure
    // In this case, a kernel launches with a grid of 1 thread blocks 
    // and each thread block has 256 parallel threads.
    add_vector<<<1, 256>>>(cuda_out, cuda_a, cuda_b, N);

    // copy result from gpu to cpu
    cudaMemcpy(out, cuda_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    //Verification
    std::cout << "result: ";
    for( int i = 0 ; i < N ; ++i )
    {
      std::cout << out[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << "vector1: ";
    for( int i = 0 ; i < N ; ++i )
    {
      std::cout << a[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << "vector2: ";
    for( int i = 0 ; i < N ; ++i )
    {
      std::cout << b[i] << "  ";
    }
    std::cout << std::endl;
    // release unused memories
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_out);
    free(a);
    free(b);
    free(out);
    return 0;
}
