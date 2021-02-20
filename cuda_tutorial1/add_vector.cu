#include <iostream>
__global__ void add_vector(float *out, float *a, float *b, int n)
{
    for( int i = 0 ; i < n ; ++i )
    {
        out[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 10;
    float *a, *b, *out;
    float *cuda_a, *cuda_b, *cuda_out;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    for( int i = 0 ; i < N ; i++ )
    {
        a[i] = i*1.1;
        b[i] = N*0.6-i;
        out[i] = 0.0;
    }
    cudaMalloc((void**)&cuda_a, sizeof(float) * N);
    cudaMalloc((void**)&cuda_b, sizeof(float) * N);
    cudaMalloc((void**)&cuda_out, sizeof(float) * N);
    cudaMemcpy(cuda_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);

    add_vector<<<1,1>>>(cuda_out, cuda_a, cuda_b, N);
    for( int i = 0 ; i < N ; ++i )
    {
      std::cout << out[i] << "  ";
    }
    std::cout << std::endl;
    cudaFree(cuda_a);
    free(a);
    return 0;
}
