void applyPermutation(double *A, int n, int *p, double* B) {
  for (int j=0;j<n;j++)
    for (int i=0;i<n;i++)
      B[j*n+i] = A[(p[j]-1)*n+p[i]-1];
}
