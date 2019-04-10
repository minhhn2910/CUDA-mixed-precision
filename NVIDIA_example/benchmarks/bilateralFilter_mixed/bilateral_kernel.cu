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

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions

#define SCATTER

#define SPEED 10

//#define SLOW_MATH

#ifdef SLOW_MATH
  #include "../../../include/cuda_math.cuh"
#else //my approximate math lib
  #include "../../../include/fast_math.cuh"
#endif

#include "newhalf.hpp"
#include <cuda_fp16.h>
__constant__ float cGaussian[64];   //gaussian array in device side
__constant__ half2 cGaussian_half[64];   //gaussian array in device side half2 type
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;

uint *dImage  = NULL;   //original image
uint *dTemp   = NULL;   //temp array for iterations
size_t pitch;

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range
    filter and domain filter, the previous one preserves crisp edges and
    the latter one filters noise. The intensity value at each pixel in
    an image is replaced by a weighted average of intensity values from
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear,
    the sample just uses a simple pixel by pixel step.

    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

    Parameters
    od - pointer to output data in global memory
    d_f - pointer to the 1D gaussian array
    e_d - euclidean delta
    w  - image width
    h  - image height
    r  - filter radius
*/

//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
}

__device__ half2 euclideanLen_half(half2_4 a, half2_4 b, half2 d)
{

    half2 mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return fast_h2exp(-mod * fast_h2rcp(__float2half2_rn(2.f) * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

//column pass using coalesced global memory reads
__global__ void
d_bilateral_filter(uint *od, int w, int h,
                   float e_d,  int r)
{
  //int block_idx = blockIdx.x*gridDim.x + blockIdx.y;
  #ifdef SCATTER
    if(blockIdx.x %10 < SPEED){ //%100 if need more resolution, this program 40 blocks so %10 is good
  #else
    if(blockIdx.x < SPEED){
  #endif

    if (threadIdx.y >= blockDim.y /2) return;

    int x = blockIdx.x*blockDim.x + threadIdx.x;

    int y1 = blockIdx.y*blockDim.y + 2*threadIdx.y;
    int y2 = blockIdx.y*blockDim.y + 2*threadIdx.y+1;

    if (x >= w || y2 >= h)
    {
        return;
    }

    half2 e_d_half = __float2half2_rn(e_d);
    half2 sum = __float2half2_rn(0.0f);
    half2 factor;
    half2_4 t = {__float2half2_rn(0.f), __float2half2_rn(0.f), __float2half2_rn(0.f), __float2half2_rn(0.f)};
    float4 center_temp1 = tex2D(rgbaTex, x, y1);
    float4 center_temp2 = tex2D(rgbaTex, x, y2);
    half2_4 center;
    center.x = __floats2half2_rn(center_temp1.x, center_temp2.x);
    center.y = __floats2half2_rn(center_temp1.y, center_temp2.y);
    center.z = __floats2half2_rn(center_temp1.z, center_temp2.z);
    //euclideanLen_half uses only x,y,z; no need to convert w;

    for (int i = -r; i <= r; i++)
    {
        for (int j = -r; j <= r; j++)
        {
            float4 curPix1 = tex2D(rgbaTex, x + j, y1 + i);
            float4 curPix2 = tex2D(rgbaTex, x + j, y2 + i);
            half2_4 curPix;
            curPix.x = __floats2half2_rn(curPix1.x, curPix2.x);
            curPix.y = __floats2half2_rn(curPix1.y, curPix2.y);
            curPix.z = __floats2half2_rn(curPix1.z, curPix2.z);
            curPix.w = __floats2half2_rn(curPix1.w, curPix2.w);

            factor = cGaussian_half[i + r] * cGaussian_half[j + r] *     //domain factor
                     euclideanLen_half(curPix, center, e_d_half);             //range factor

            t.x += factor * curPix.x;
            t.y += factor * curPix.y;
            t.z += factor * curPix.z;
            t.w += factor * curPix.w;
            sum += factor;
        }
    }
    float4 res1, res2;
    half2_4 res_half;
    half2 rcp_sum = fast_h2rcp(sum);
    res_half.x = t.x*rcp_sum;
    res_half.y = t.y*rcp_sum;
    res_half.z = t.z*rcp_sum;
    res_half.w = t.w*rcp_sum;
    float2 temp = __half22float2(res_half.x);
    res1.x = temp.x;
    res2.x = temp.y;

    temp = __half22float2(res_half.y);
    res1.y = temp.x;
    res2.y = temp.y;

    temp = __half22float2(res_half.z);
    res1.z = temp.x;
    res2.z = temp.y;

    temp = __half22float2(res_half.w);
    res1.w = temp.x;
    res2.w = temp.y;

    od[y1 * w + x] = rgbaFloatToInt(res1);
    od[y2 * w + x] = rgbaFloatToInt(res2);

  }
  else {
    //doing float
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float4 t = {0.f, 0.f, 0.f, 0.f};
    float4 center = tex2D(rgbaTex, x, y);

    for (int i = -r; i <= r; i++)
    {
        for (int j = -r; j <= r; j++)
        {
            float4 curPix = tex2D(rgbaTex, x + j, y + i);
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                     euclideanLen(curPix, center, e_d);             //range factor

            t += factor * curPix;
            sum += factor;
        }
    }

    od[y * w + x] = rgbaFloatToInt(t/sum);
  }

}

extern "C"
void initTexture(int width, int height, uint *hImage)
{
    // copy image data to array
    checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMallocPitch(&dTemp,  &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint)*width,
                                 sizeof(uint)*width, height, cudaMemcpyHostToDevice));
}

extern "C"
void freeTextures()
{
    checkCudaErrors(cudaFree(dImage));
    checkCudaErrors(cudaFree(dTemp));
}

/*
    Because a 2D gaussian mask is symmetry in row and column,
    here only generate a 1D mask, and use the product by row
    and column index later.

    1D gaussian distribution :
        g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

    parameters:
    og - output gaussian array in global memory
    delta - the 2nd parameter 'd' in the above function
    radius - half of the filter size
             (total filter size = 2 * radius + 1)
*/
extern "C"
void updateGaussian(float delta, int radius)
{
    float  fGaussian[64];
    uint32_t half2Gaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));

        half2Gaussian[i] =  floats2half2 (fGaussian[i],fGaussian[i]);
    }
    //uint32_t floats2half2 (float val_high, float val_low )
    checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1)));
    checkCudaErrors(cudaMemcpyToSymbol(cGaussian_half, half2Gaussian, sizeof(uint32_t)*(2*radius+1)));
}

/*
    Perform 2D bilateral filter on image using CUDA

    Parameters:
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    e_d    - euclidean delta
    radius - filter radius
    iterations - number of iterations
*/

// RGBA version
extern "C"
double bilateralFilterRGBA(uint *dDest,
                           int width, int height,
                           float e_d, int radius, int iterations,
                           StopWatchInterface *timer)
{
    // var for kernel computation timing
    double dKernelTime;

    // Bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, width, height, pitch));

    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filter<<< gridSize, blockSize>>>(
            dDest, width, height, e_d, radius);

        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
                                         sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}
