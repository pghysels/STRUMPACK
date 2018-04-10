double * applyPermutation(double *A, int n, int *p) {
  double *B;
  int i,j;

  B=new double[n*n];
  for(j=0;j<n;j++)
    for(i=0;i<n;i++)
      B[j*n+i]=A[(p[j]-1)*n+p[i]-1];

  return B;
}
