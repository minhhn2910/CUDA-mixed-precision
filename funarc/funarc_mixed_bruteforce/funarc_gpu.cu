#include <iostream>
#include <math.h>
#include <stdio.h>
#define TYPE1 double
#define TYPE2 float
#define TYPE3 float
#define TYPE4 double
#define SPEED 0
double fun_ref( double x){
  int k, n = 5;
  double t1;
  double d1 = 1.0;
  t1 = x;
  for ( k = 1; k <= n; k++ )
    {
      d1 = 2.0 * d1;
      t1 = t1+ sin(d1 * x)/d1;
    }
    return t1;
}
__global__ void fun_gpu(double x[], double y[], int nthreads, int speed){
  //y = fun(x)
  //speed = %
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(blockIdx.x % 10< speed){
    int k, n = 5;
    if (tid < nthreads) {
      double t1;
      float d1 = 1.0;
      float x_temp = x[tid];
      t1 = x[tid];
      for ( k = 1; k <= n; k++ )
        {
          d1 = 2.0 * d1;
          double sin_res = sin(d1 * x_temp);
          t1 = t1 + sin_res/d1;
        }
      y[tid] = t1;
  }

  }
  else{

        int k, n = 5;
        if (tid < nthreads) {
          double t1;
          double d1 = 1.0;
          double x_temp = x[tid];
          t1 = x[tid];
          for ( k = 1; k <= n; k++ )
            {
              d1 = 2.0 * d1;
              double sin_res = sin(d1 * x_temp);
              t1 = t1 + sin_res/d1;
            }
          y[tid] = t1;
        }
 }

}


int main( int argc, char **argv) {
  int i,n = 1000000;
  double h, t1, t2, dppi;
  double s1;
  //cuda def
  cudaEvent_t start, stop;
  float elapsedTime;
  int speed = atoi(argv[1]);
  printf("running with speed %d \n", speed);
  double *d_x, *d_y, *h_x, *h_y ;
  size_t size = n*sizeof(double);

  h_x = (double*) malloc(size);
  h_y = (double*) malloc(size);
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);


  t1 = -1.0;
  dppi = acos(t1);
  s1 = 0.0;
  t1 = 0.0;
  h = dppi / n;
  for ( i = 1; i <= n; i++){
    h_x[i-1] = i * h;
  }
    /* Copy vectors from host memory to device memory */
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

  int threads_per_block = 256;

  int block_count = (n + threads_per_block - 1)/threads_per_block;
  cudaEventCreate(&start);
  cudaEventRecord(start,0);
  for (int i =0;i < 10; i ++)
    fun_gpu<<<block_count, threads_per_block>>>(d_x, d_y, n,speed);

  cudaDeviceSynchronize();
  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time : %f ms\n" ,elapsedTime);
  cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
  for ( i = 1; i <= n; i++)
    {
      t2 = h_y[i-1];
      s1 = s1 + sqrt(h*h + (t2 - t1) * (t2 - t1));
      t1 = t2;
    }
  double ref_value = 5.7957763224;
  printf("%.10f\n",s1);
  printf("abs err %.8f  rel err %.8f\n", fabs(s1-ref_value), fabs((s1-ref_value)/ref_value) );
  return 0;
}
