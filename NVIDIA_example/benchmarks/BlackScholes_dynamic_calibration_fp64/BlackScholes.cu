/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include <cuda_profiler_api.h>

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 10000000;
const int  NUM_ITERATIONS = 100;
#define LEN_N 150

const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
bool is_calibration_iter (int n){
  return (n<7) || n ==16 || n==36 || n==66 || (n >= 101 && n<=107) || n== 117 || n== 137;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);

    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    float
        *h_CallResultGPU_fp64,
        *h_PutResultGPU_fp64;

    //reading speed for searching
    int speed = atoi(argv[1]);
    printf("running with speed %d", speed);

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    StopWatchInterface *hTimer = NULL;
    int i;

    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);


    h_CallResultGPU_fp64 = (float *)malloc(OPT_SZ);
    h_PutResultGPU_fp64 = (float *)malloc(OPT_SZ);

    printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));
    //warm up for performance
    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP((OPT_N/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
            0
        );
        getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    printf("...generating input data in CPU mem.\n");
    int iteration = 0;

    double accumulate_runtime = 0.0;
    double accumulate_option = 0.0;

    double runtimes[LEN_N];
    double error[LEN_N];

    int lower_bound = 0;
    int upper_bound = 100;
    int mid = 50;
    double error_threshold = 7E-08;
    bool calibration = true;
for (iteration = 0 ; iteration <10; iteration ++){
    srand(iteration);
    if (iteration == 101 ) //reset calibration
    {
       lower_bound = 0;
       upper_bound = 100;
       mid = 50;
       error_threshold = 1.4E-7;
    }
    calibration = is_calibration_iter(iteration);
    //Generate options set
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }
    //binary search
    mid = (upper_bound + lower_bound)/2;

//    printf(" low %d up %d mid %d \n", lower_bound, upper_bound, (upper_bound + lower_bound)/2 );
    printf("\n mid calibration %d \n", mid);
    //
//    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));
//    printf("Data init done.\n\n");

    runtimes[iteration] = 0;

//if (iteration< 7) calibration = true;
// if(calibration){ //calibration run
//  printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (i = 0; i < NUM_ITERATIONS; i++)
  {
      BlackScholesGPU<<<DIV_UP((OPT_N/2), 128), 128/*480, 128*/>>>(
          (float2 *)d_CallResult,
          (float2 *)d_PutResult,
          (float2 *)d_StockPrice,
          (float2 *)d_OptionStrike,
          (float2 *)d_OptionYears,
          RISKFREE,
          VOLATILITY,
          OPT_N,
          0
      );
      getLastCudaError("BlackScholesGPU() execution failed\n");
  }

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;
  if( calibration)
  {
    runtimes[iteration] = gpuTime;
    accumulate_runtime += gpuTime;
  } else
    runtimes[iteration] = 0;

  accumulate_option +=  2 * OPT_N;
  //Both call and put is calculated
//  printf("Options count             : %i     \n", 2 * OPT_N);
  printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
//  printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
 printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

//  printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
//         (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

//  printf("\nReading back GPU results...\n");
  //Read back GPU results to compare them to CPU results
  checkCudaErrors(cudaMemcpy(h_CallResultGPU_fp64, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_PutResultGPU_fp64,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));


//  printf("Checking the results...\n");
//  printf("...running CPU calculations.\n\n");
  //Calculate options values on CPU
/*  BlackScholesCPU(
      h_CallResultCPU,
      h_PutResultCPU,
      h_StockPrice,
      h_OptionStrike,
      h_OptionYears,
      RISKFREE,
      VOLATILITY,
      OPT_N
  );
*/
//  printf("Comparing the results...\n");
  //Calculate max absolute difference and L1 distance
  //between CPU and GPU results
  sum_delta = 0;
  sum_ref   = 0;
  max_delta = 0;

  for (i = 0; i < OPT_N; i++)
  {
      ref   = h_CallResultGPU_fp64[i];
      delta = fabs(h_CallResultGPU_fp64[i] - h_CallResultGPU_fp64[i]);

      if (delta > max_delta)
      {
          max_delta = delta;
      }

      sum_delta += delta;
      sum_ref   += fabs(ref);
  }

  L1norm = sum_delta / sum_ref;

//}


//    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP((OPT_N/2), 128), 128/*480, 128*/>>>(
            (float2 *)d_CallResult,
            (float2 *)d_PutResult,
            (float2 *)d_StockPrice,
            (float2 *)d_OptionStrike,
            (float2 *)d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
            mid
        );
        getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;


    //Both call and put is calculated
//    printf("Options count             : %i     \n", 2 * OPT_N);
//    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
//    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

//    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
//           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 128);

//    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));


//    printf("Checking the results...\n");
//    printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
/*    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );
*/
//    printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultGPU_fp64[i];
        delta = fabs(h_CallResultGPU_fp64[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    //repeat wit appropriate token for searching script
    printf("tuning_error=%E\n", L1norm);
//    printf("Max absolute error: %E\n\n", max_delta);

    printf("err error_threshold: %E\n", error_threshold);

        runtimes[iteration] += gpuTime;
        if(!calibration)
          error[iteration] = L1norm;
        else
          error[iteration] = 0;

        accumulate_runtime += gpuTime;
        accumulate_option +=  2 * OPT_N;


    if (L1norm < error_threshold){
      lower_bound = mid;
    } else {
      upper_bound = mid;
    }
    if( upper_bound - lower_bound <= 2)
      calibration = false;





} //end for iteration

    printf("\n gputime \n");
    for (int i =0; i<LEN_N ; i++){
      printf("%f, ", runtimes[i]);
    }
    printf("\n error \n");
    for (int i =0; i<LEN_N ; i++){
      printf("%.4E, ", error[i]);
    }
    printf("\n accumulate runtime %f , accumulate_opt %E \n", accumulate_runtime, accumulate_option);


    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

    printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    sdkDeleteTimer(&hTimer);
    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");

    // Calling cudaProfilerStop causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaProfilerStop());

    if (L1norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
