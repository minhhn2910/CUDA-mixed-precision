#include <iostream>
#include <math.h>
#include <stdio.h>
#include "cuda_fp16.h"
#include "../../include/fast_math.cuh"

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
__global__ void fun_gpu(float x[], float y[], int nthreads){
  //y = fun(x)
  if (threadIdx.x > blockDim.x/2)
      return; //50\% the warps in each block are inactive.
  int tid1 = blockDim.x * blockIdx.x + 2*threadIdx.x;
  int tid2 = blockDim.x * blockIdx.x + 2*threadIdx.x+1;
  int k, n = 5;
  if (tid2 < nthreads) {
    half2 t1;
    half2 d1 = __float2half2_rn(1.0);
    half2 x_temp = __floats2half2_rn(x[tid1], x[tid2]);
    t1 = x_temp;
    for ( k = 1; k <= n; k++ )
    {
        d1 = 2.0 * d1;
        half2 sin_res = fast_h2sin(d1 * x_temp);
        t1 = t1 + sin_res*fast_h2rcp(d1);
    }

    float2 y_temp = __half22float2(t1);//convert back to float
    y[tid1] = y_temp.x;
    y[tid2] = y_temp.y;
  }
}


int main( int argc, char **argv) {
  int i,n = 1000000;
  double h, t1, t2, dppi;
  double s1;
  //cuda def
  cudaEvent_t start, stop;
  float elapsedTime;
  float *d_x, *d_y, *h_x, *h_y ;
  size_t size = n*sizeof(float);

  h_x = (float*) malloc(size);
  h_y = (float*) malloc(size);
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
    fun_gpu<<<block_count, threads_per_block>>>(d_x, d_y, n);

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
      if (i==10)
      printf("%f %f \n", t2, s1);
      t1 = t2;
    }
  double ref_value = 5.7957763224;
  printf("%.10f\n",s1);
  printf("abs err %.8f  rel err %.8f\n", fabs(s1-ref_value), fabs((s1-ref_value)/ref_value) );
  return 0;
}
