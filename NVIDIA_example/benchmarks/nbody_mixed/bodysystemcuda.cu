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

//#include "../../../include/fast_math.cuh"
#define SLOW_MATH

#ifdef SLOW_MATH
  #include "../../../include/cuda_math.cuh"
#else //my approximate math lib
  #include "../../../include/fast_math.cuh"
#endif

#include <helper_cuda.h>
#include <math.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "bodysystem.h"

#define SCATTER

#define SPEED 10
/*
typedef struct __device_builtin__ half2_3
{
    half2 x, y, z;
} half2_3 ;

typedef struct __device_builtin__ __builtin_align__(16) half2_4
{
    half2 x, y, z, w;
} half2_4;
*/
//typedef __device_builtin__ struct half2_3 half2_3;
//typedef __device_builtin__ struct half2_4 half2_4;
__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared,
                              &softeningSq,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64,
                              &softeningSq,
                              sizeof(double), 0,
                              cudaMemcpyHostToDevice);
}

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<typename T>
__device__ T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}

template<>
__device__ double rsqrt_T<double>(double x)
{
    return rsqrt(x);
}


// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

template <typename T>
__device__ T getSofteningSquared()
{
    return softeningSquared;
}
template <>
__device__ double getSofteningSquared<double>()
{
    return softeningSquared_fp64;
}

template <typename T>
struct DeviceData
{
    T *dPos[2]; // mapped host pointers
    T *dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};


template <typename T>
__device__ typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai,
                    typename vec4<T>::Type bi,
                    typename vec4<T>::Type bj)
{
    typename vec3<T>::Type r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = rsqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ half2_3
bodyBodyInteraction_half2( half2_3 ai,
                     half2_4 bi,
                     half2_4 bj)
{
     half2_3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    half2 distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += __float2half2_rn(getSofteningSquared<float>());

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    half2 invDist = fast_h2rsqrt(distSqr);
    half2 invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    half2 s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}


template <typename T>
__device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos,
                 typename vec4<T>::Type *positions,
                 int numTiles, cg::thread_block cta)
{
    typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        cg::sync(cta);

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
        }

        cg::sync(cta);
    }

    return acc;
}


__device__ half2_3
computeBodyAccel_half2(half2_4 bodyPos,
                 typename vec4<float>::Type *positions,
                 int numTiles, cg::thread_block cta)
{
    half2_4 *sharedPos = SharedMemory<half2_4>();

    half2_3 acc;// = {0.0f, 0.0f, 0.0f};
    acc.x = __float2half2_rn(0.0);
    acc.y = __float2half2_rn(0.0);
    acc.z = __float2half2_rn(0.0);
    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[2*threadIdx.x].x = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x].x);
	    	sharedPos[2*threadIdx.x].y = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x].y);
	    	sharedPos[2*threadIdx.x].z = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x].z);
	    	sharedPos[2*threadIdx.x].w = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x].w);

        sharedPos[2*threadIdx.x+1].x = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x+1].x);
	    	sharedPos[2*threadIdx.x+1].y = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x+1].y);
	    	sharedPos[2*threadIdx.x+1].z = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x+1].z);
	    	sharedPos[2*threadIdx.x+1].w = __float2half2_rn(positions[tile * blockDim.x + 2*threadIdx.x+1].w);

        cg::sync(cta);

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction_half2(acc, bodyPos, sharedPos[counter]);
        }

        cg::sync(cta);
    }

    return acc;
}

template<typename T>
__global__ void
integrateBodies(typename vec4<T>::Type *__restrict__ newPos,
                typename vec4<T>::Type *__restrict__ oldPos,
                typename vec4<T>::Type *vel,
                unsigned int deviceOffset, unsigned int deviceNumBodies,
                float deltaTime, float damping, int numTiles)
{

#ifdef SCATTER
  if( blockIdx.x %10 < SPEED){ //speed from 1- 10 .  blockIdx.x %10 < SPEED if need more resolution.
#else
  if( blockIdx.x < SPEED){
#endif
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    if (threadIdx.x >= blockDim.x/2)
    {
        return;
    }
    int index = blockIdx.x * blockDim.x + 2*threadIdx.x;
    int index1 = blockIdx.x * blockDim.x + 2*threadIdx.x+1;


//    typename vec4<T>::Type position = oldPos[deviceOffset + index];

	 half2_4 position;
	  position.x = __floats2half2_rn(oldPos[deviceOffset + index].x,oldPos[deviceOffset + index1].x);
	   position.y = __floats2half2_rn(oldPos[deviceOffset + index].y,oldPos[deviceOffset + index1].y);
	    position.z = __floats2half2_rn(oldPos[deviceOffset + index].z,oldPos[deviceOffset + index1].z);
	     position.w = __floats2half2_rn(oldPos[deviceOffset + index].w,oldPos[deviceOffset + index1].w);
  /*
  if(index==0){
      printf("\device \n");
      printf("%f %f %f %f \n", __half2float(position.x.x),__half2float(position.y.x),__half2float(position.z.x),__half2float(position.w.x) );
      printf("%f %f %f %f \n", __half2float(position.x.y),__half2float(position.y.y),__half2float(position.z.y),__half2float(position.w.y) );

    }
    */
   // typename vec3<T>::Type accel = computeBodyAccel<T>(position, oldPos, numTiles, cta);

   half2_3 accel = computeBodyAccel_half2(position,
                                                       oldPos,
                                                       numTiles, cta);


    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    // typename vec4<T>::Type velocity = vel[deviceOffset + index];
	   half2_4 velocity ;
	    velocity.x = __floats2half2_rn(vel[deviceOffset + index].x,vel[deviceOffset + index1].x);
	     velocity.y = __floats2half2_rn(vel[deviceOffset + index].y,vel[deviceOffset + index1].y);
	      velocity.z = __floats2half2_rn(vel[deviceOffset + index].z,vel[deviceOffset + index1].z);
	       velocity.w = __floats2half2_rn(vel[deviceOffset + index].w,vel[deviceOffset + index1].w);
/*
  if(index==0){
      printf("\device velocity\n");
      printf("%f %f %f %f \n", __half2float(velocity.x.x),__half2float(velocity.y.x),__half2float(velocity.z.x),__half2float(velocity.w.x) );
      printf("%f %f %f %f \n", __half2float(velocity.x.y),__half2float(velocity.y.y),__half2float(velocity.z.y),__half2float(velocity.w.y) );

    }
*/
    velocity.x += accel.x * __float2half2_rn(deltaTime);
    velocity.y += accel.y * __float2half2_rn(deltaTime);
    velocity.z += accel.z * __float2half2_rn(deltaTime);

    velocity.x *= __float2half2_rn(damping);
    velocity.y *= __float2half2_rn(damping);
    velocity.z *= __float2half2_rn(damping);

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * __float2half2_rn(deltaTime);
    position.y += velocity.y * __float2half2_rn(deltaTime);
    position.z += velocity.z * __float2half2_rn(deltaTime);

    // store new position and velocity
    //newPos[deviceOffset + index] = position;
    //vel[deviceOffset + index]    = velocity;
    float2 temp = __half22float2(position.x);
    newPos[deviceOffset + index].x = temp.x;
    newPos[deviceOffset + index1].x = temp.y;

 	  temp = __half22float2(position.y);
    newPos[deviceOffset + index].y = temp.x;
    newPos[deviceOffset + index1].y = temp.y;

    temp = __half22float2(position.z);
    newPos[deviceOffset + index].z = temp.x;
    newPos[deviceOffset + index1].z = temp.y;

    temp = __half22float2(position.w);
    newPos[deviceOffset + index].w = temp.x;
    newPos[deviceOffset + index1].w = temp.y;

    temp = __half22float2(velocity.x);
    vel[deviceOffset + index].x = temp.x;
    vel[deviceOffset + index1].x = temp.y;

    temp = __half22float2(velocity.y);
    vel[deviceOffset + index].y = temp.x;
    vel[deviceOffset + index1].y = temp.y;


    temp = __half22float2(velocity.z);
    vel[deviceOffset + index].z = temp.x;
    vel[deviceOffset + index1].z = temp.y;


    temp = __half22float2(velocity.w);
    vel[deviceOffset + index].w = temp.x;
    vel[deviceOffset + index1].w = temp.y;


  } else {
    //if(blockIdx.x %10 == 0 && threadIdx.x ==0)
    //  printf("doing float %d\n", blockIdx.x );
    //doing float comp
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies)
    {
        return;
    }

    typename vec4<T>::Type position = oldPos[deviceOffset + index];

    typename vec3<T>::Type accel = computeBodyAccel<T>(position,
                                                       oldPos,
                                                       numTiles, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    typename vec4<T>::Type velocity = vel[deviceOffset + index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[deviceOffset + index] = position;
    vel[deviceOffset + index]    = velocity;

  }
}

template <typename T>
void integrateNbodySystem(DeviceData<T> *deviceData,
                          cudaGraphicsResource **pgres,
                          unsigned int currentRead,
                          float deltaTime,
                          float damping,
                          unsigned int numBodies,
                          unsigned int numDevices,
                          int blockSize,
                          bool bUsePBO)
{
    if (bUsePBO)
    {
        checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[currentRead], cudaGraphicsMapFlagsReadOnly));
        checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[1-currentRead], cudaGraphicsMapFlagsWriteDiscard));
        checkCudaErrors(cudaGraphicsMapResources(2, pgres, 0));
        size_t bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[currentRead]), &bytes, pgres[currentRead]));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[1-currentRead]), &bytes, pgres[1-currentRead]));
    }

    for (unsigned int dev = 0; dev != numDevices; dev++)
    {
        if (numDevices > 1)
        {
            cudaSetDevice(dev);
        }

        int numBlocks = (deviceData[dev].numBodies + blockSize-1) / blockSize;
        int numTiles = (numBodies + blockSize - 1) / blockSize;
        int sharedMemSize = blockSize * 4 * sizeof(T); // 4 floats for pos
/*
        integrateBodies<T><<< numBlocks, blockSize, sharedMemSize >>>
        ((typename vec4<T>::Type *)deviceData[dev].dPos[1-currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dVel,
         deviceData[dev].offset, deviceData[dev].numBodies,
         deltaTime, damping, numTiles);
    */
        integrateBodies<float><<< numBlocks, blockSize, sharedMemSize >>>
        ((typename vec4<float>::Type *)deviceData[dev].dPos[1-currentRead],
         (typename vec4<float>::Type *)deviceData[dev].dPos[currentRead],
         (typename vec4<float>::Type *)deviceData[dev].dVel,
         deviceData[dev].offset, deviceData[dev].numBodies,
         deltaTime, damping, numTiles);

        if (numDevices > 1)
        {
            checkCudaErrors(cudaEventRecord(deviceData[dev].event));
            // MJH: Hack on older driver versions to force kernel launches to flush!
            cudaStreamQuery(0);
        }

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    if (numDevices > 1)
    {
        for (unsigned int dev = 0; dev < numDevices; dev++)
        {
            checkCudaErrors(cudaEventSynchronize(deviceData[dev].event));
        }
    }

    if (bUsePBO)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(2, pgres, 0));
    }
}


// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(DeviceData<float> *deviceData,
                                          cudaGraphicsResource **pgres,
                                          unsigned int currentRead,
                                          float deltaTime,
                                          float damping,
                                          unsigned int numBodies,
                                          unsigned int numDevices,
                                          int blockSize,
                                          bool bUsePBO);

template void integrateNbodySystem<double>(DeviceData<double> *deviceData,
                                           cudaGraphicsResource **pgres,
                                           unsigned int currentRead,
                                           float deltaTime,
                                           float damping,
                                           unsigned int numBodies,
                                           unsigned int numDevices,
                                           int blockSize,
                                           bool bUsePBO);
