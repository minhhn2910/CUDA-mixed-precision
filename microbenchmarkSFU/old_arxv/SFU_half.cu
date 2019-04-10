#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "cuda_fp16.h"
#include "fast_math.cuh"
#define L1_SIZE 65536
#define FP_TYPE float
#define FP_DEV_TYPE float
/* Kernel for vector addition */

__global__ void Vec_add(FP_DEV_TYPE x[], FP_DEV_TYPE y[], FP_DEV_TYPE z[], int n,  FP_DEV_TYPE lookup[], uint32_t startClk[], uint32_t stopClk[]) {
   /* blockDim.x = threads_per_block                            */
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   /* block_count*threads_per_block may be >= n */
   // a register to avoid compiler optimization
   half sink = 0;


   if (tid < n) {

     half temp = __float2half_rn(x[tid]);
     // synchronize all threads
     asm volatile ("bar.sync 0;");
     uint32_t start = 0;
    // start timing
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

// doing computation

    sink = hexp(temp);
    // synchronize all threads
    //asm volatile ("bar.sync 0;");
    // stop timing
    uint32_t stop = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    // write time and data back to memory
    startClk[tid] = start;
    stopClk[tid] = stop;
  //  dsink[tid] = sink;

    z[tid] = sink;
   }
}  /* Vec_add */


/* Host code */
int main(int argc, char* argv[]) {

   int n, i;
   FP_TYPE *h_x, *h_y, *h_z, *h_lookup;
   FP_DEV_TYPE *d_x, *d_y, *d_z, *d_lookup ;
   uint32_t *h_startClk, *h_stopClk;
   uint32_t *d_startClk, *d_stopClk;
   int threads_per_block;
   int block_count;
   size_t size, size_clock;
	cudaEvent_t start, stop;
  float elapsedTime;
  srand(time(0));
   /* Get number of components in vector */
   if (argc != 2) {
      fprintf(stderr, "usage: %s <vector order>\n", argv[0]);
      exit(0);
   }
   n = strtol(argv[1], NULL, 10); // half2 = 2x half , reduce size
   size = n*sizeof(FP_TYPE);
   size_clock = n*sizeof(uint32_t);
   /* Allocate input vectors in host memory */
   h_x = (FP_TYPE*) malloc(size);
   h_y = (FP_TYPE*) malloc(size);
   h_z = (FP_TYPE*) malloc(size);

   h_startClk = (uint32_t*) malloc(size_clock);
   h_stopClk = (uint32_t*) malloc(size_clock);

   h_lookup = (FP_TYPE*) malloc(L1_SIZE*sizeof(FP_TYPE));
   // declare and allocate memory

   /* Initialize input vectors */
   for (i = 0; i < n; i++) {
     h_x[i] = 1.0/i;
  //    h_x[i] = rand()%L1_SIZE;
   }
   for (i=0;i<L1_SIZE;i++)
      h_lookup[i] = (i*5)%L1_SIZE;

   /* Allocate vectors in device memory */
   cudaMalloc(&d_x, size);
   cudaMalloc(&d_y, size);
   cudaMalloc(&d_z, size);
   cudaMalloc(&d_lookup, L1_SIZE*sizeof(FP_TYPE));
   cudaMalloc(&d_stopClk, size_clock);
   cudaMalloc(&d_startClk, size_clock);

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_lookup, h_lookup, L1_SIZE*sizeof(FP_TYPE), cudaMemcpyHostToDevice);

//   cudaMemcpy(buffer, h_buffer, MAX_TEXTURE_SIZE*sizeof(float), cudaMemcpyHostToDevice); //copy data to texture

   /* Define block size */
   threads_per_block = 256;

   block_count = (n + threads_per_block - 1)/threads_per_block;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

   Vec_add<<<block_count, threads_per_block>>>(d_x, d_y, d_z, n, d_lookup, d_startClk,d_stopClk);

   cudaThreadSynchronize();
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
 cudaEventElapsedTime(&elapsedTime, start,stop);
 printf("Elapsed time : %f ms\n" ,elapsedTime);
  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_startClk, d_startClk, size_clock, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_stopClk, d_stopClk, size_clock, cudaMemcpyDeviceToHost);

uint32_t sum = 0;
printf("clk cycles spent on each thread \n");
for (i = 0; i < n; i++) {
;
//  printf("%d, \n", h_z[i]);
  printf("%u,", h_stopClk[i] - h_startClk[i]);
  sum+=h_stopClk[i] - h_startClk[i];
}


printf("\n -------- \n average latency (cycles) %f \n",float(sum)/n);
   /* Free device memory */
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_z);

   /* Free host memory */
   free(h_x);
   free(h_y);
   free(h_z);

   return 0;
}  /* main */
