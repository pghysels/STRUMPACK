#include <iostream>

int main() {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
  for (int i=0; i<100; i++)
    std::cout << "hello" << std::endl;
  return 0;
}
