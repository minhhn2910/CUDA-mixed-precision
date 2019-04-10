#include <iostream>
#include <math.h>
#include <stdio.h>
#include "cuda_fp16.h"
#include "../../include/fast_math.cuh"
float fun_ref( float x){
  int k, n = 5;
  float t1;
  float d1 = 1.0;
  t1 = x;
  for ( k = 1; k <= n; k++ )
    {
      d1 = 2.0 * d1;
      t1 = t1+ sinf(d1 * x)/d1;
    }
    return t1;
}
__global__ void fun_gpu(float x[], float y[], int nthreads, int speed){
  //y = fun(x)
  if(blockIdx.x %10 < speed){
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
          t1 = t1 + sin_res * fast_h2rcp(d1);
      }
    float2 y_temp = __half22float2(t1);//convert back to float
    y[tid1] = y_temp.x;
    y[tid2] = y_temp.y;
  }
 }else{
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   int k, n = 5;
   if (tid < nthreads) {
     float t1;
     float d1 = 1.0;
     float x_temp = x[tid];
     t1 = x[tid];
       for ( k = 1; k <= n; k++ )
         {
           d1 = 2.0 * d1;
           float sin_res = sinf(d1 * x_temp);
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
  float *d_x, *d_y, *h_x, *h_y ;
  size_t size = n*sizeof(float);

  h_x = (float*) malloc(size);
  h_y = (float*) malloc(size);
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);
  int speed = atoi(argv[1]);
  printf("running with speed %d \n", speed);
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

  int threads_per_block = 512;

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
      //if (i==900000)
      //printf("%f %f \n", t2, s1);
      t1 = t2;
    }
    //rel error
    double error = 0.0;
    for ( i = 1; i <= n; i++)
    {
      float t2_ref = fun_ref(i * h);
      double temp_err;
      if (t2_ref != 0)
        temp_err = fabs((h_y[i-1] - t2_ref)/t2_ref);
      else
        temp_err = fabs(h_y[i-1]);

      error += temp_err;
    }
    printf("avg rel error %f \n",error/n);

  return 0;
}
