#include <iostream>

int main() {
  int x=0, y=5;
#pragma omp parallel
#pragma omp single
  {
#pragma omp task depend(inout:x) depend(out:y) priority(5)
    {
      y = x;
      x += 3;
    }
#pragma omp task depend(in:x) depend(out:y) priority(1)
    {
      y = 3 * x;
    }
  }
  return 0;
}
