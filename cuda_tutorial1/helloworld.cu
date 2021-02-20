#include <iostream>
__global__ void hello()
{
  std::cout << "Hello World from GPU!" << std::endl;
}

int main()
{
    hello<<<1,1>>>();
    return 0;
}
