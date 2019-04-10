/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "cuenergy.h"
#include "half2_operator_overload.cuh"
//#include "fast_math.cuh"

//#define SLOW_MATH
//#define SCATTER

#ifdef SLOW_MATH
  #include "cuda_math.cuh"
#else //my approximate math lib
  #include "fast_math.cuh"
#endif

#include "cuda_fp16.h"
#include <stdio.h>

#if UNROLLX != 8
# error "UNROLLX must be 8"
#endif

#if BLOCKSIZEX != 16
# error "BLOCKSIZEX must be 16"
#endif

#define SPEED 127
// Max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about.
// At 16 bytes for atom, for this program 4070 atoms is about the max
// we can store in the constant buffer.
__constant__ float4 atominfo[MAXATOMS];

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.

__global__ void cenergy(int numatoms, float gridspacing, float * energygrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
                         + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
                         + xindex;

#ifdef SCATTER
  if(blockIdx.y %10 < SPEED){ //%100 if need more resolution, this program 127 blocks so %10 is good
#else
  if(blockIdx.y < SPEED){
 #endif

//      if(threadIdx.x == 0 && threadIdx.y==0)
//        printf("half running \n");
  half2 coory = __float2half2_rn(gridspacing * yindex);
  half2 coorx = __float2half2_rn(gridspacing * xindex);

  half2 energyvalx1= __float2half2_rn(0.0f);
  half2 energyvalx2= __float2half2_rn(0.0f);
  half2 energyvalx3= __float2half2_rn(0.0f);
  half2 energyvalx4= __float2half2_rn(0.0f);
  half2 energyvalx5= __float2half2_rn(0.0f);
  half2 energyvalx6= __float2half2_rn(0.0f);
  half2 energyvalx7= __float2half2_rn(0.0f);
  half2 energyvalx8= __float2half2_rn(0.0f);

  half2 gridspacing_u = __float2half2_rn(gridspacing * BLOCKSIZEX);

  int atomid;
  for (atomid=0; atomid<numatoms/2; atomid++) {
    half2 atom_info_y = __floats2half2_rn(atominfo[2*atomid].y, atominfo[2*atomid+1].y);
    //float dy = coory - atominfo[atomid].y;
    //float dyz2 = (dy * dy) + atominfo[atomid].z;
    half2 atom_info_z = __floats2half2_rn(atominfo[2*atomid].z, atominfo[2*atomid+1].z);
    half2 atom_info_x = __floats2half2_rn(atominfo[2*atomid].x, atominfo[2*atomid+1].x);
    half2 atom_info_w = __floats2half2_rn(atominfo[2*atomid].w, atominfo[2*atomid+1].w);

    half2 dy = coory - atom_info_y;
    half2 dyz2 = (dy * dy) + atom_info_z;
    //float dx1 = coorx - atominfo[atomid].x;
    half2 dx1 = coorx - atom_info_x;
    half2 dx2 = dx1 + gridspacing_u;
    half2 dx3 = dx2 + gridspacing_u;
    half2 dx4 = dx3 + gridspacing_u;
    half2 dx5 = dx4 + gridspacing_u;
    half2 dx6 = dx5 + gridspacing_u;
    half2 dx7 = dx6 + gridspacing_u;
    half2 dx8 = dx7 + gridspacing_u;

    energyvalx1 += atom_info_w * (fast_h2rsqrt(dx1*dx1 + dyz2));
    energyvalx2 += atom_info_w * (fast_h2rsqrt(dx2*dx2 + dyz2));
    energyvalx3 += atom_info_w * (fast_h2rsqrt(dx3*dx3 + dyz2));
    energyvalx4 += atom_info_w * (fast_h2rsqrt(dx4*dx4 + dyz2));
    energyvalx5 += atom_info_w * (fast_h2rsqrt(dx5*dx5 + dyz2));
    energyvalx6 += atom_info_w * (fast_h2rsqrt(dx6*dx6 + dyz2));
    energyvalx7 += atom_info_w * (fast_h2rsqrt(dx7*dx7 + dyz2));
    energyvalx8 += atom_info_w * (fast_h2rsqrt(dx8*dx8 + dyz2));
  }

  float energyvalx1_float = __half2float(energyvalx1.x + energyvalx1.y);
  float energyvalx2_float = __half2float(energyvalx2.x + energyvalx2.y);
  float energyvalx3_float = __half2float(energyvalx3.x + energyvalx3.y);
  float energyvalx4_float = __half2float(energyvalx4.x + energyvalx4.y);
  float energyvalx5_float = __half2float(energyvalx5.x + energyvalx5.y);
  float energyvalx6_float = __half2float(energyvalx6.x + energyvalx6.y);
  float energyvalx7_float = __half2float(energyvalx7.x + energyvalx7.y);
  float energyvalx8_float = __half2float(energyvalx8.x + energyvalx8.y);

  energygrid[outaddr]   += energyvalx1_float;
  energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2_float;
  energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3_float;
  energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4_float;
  energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5_float;
  energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6_float;
  energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7_float;
  energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8_float;

}else{
   //doing float computation

//    if(threadIdx.x == 0 && threadIdx.y==0)
//     printf("float running \n");
     float coory = gridspacing * yindex;
     float coorx = gridspacing * xindex;

     float energyvalx1=0.0f;
     float energyvalx2=0.0f;
     float energyvalx3=0.0f;
     float energyvalx4=0.0f;
     float energyvalx5=0.0f;
     float energyvalx6=0.0f;
     float energyvalx7=0.0f;
     float energyvalx8=0.0f;

     float gridspacing_u = gridspacing * BLOCKSIZEX;

     int atomid;
     for (atomid=0; atomid<numatoms; atomid++) {
       float dy = coory - atominfo[atomid].y;
       float dyz2 = (dy * dy) + atominfo[atomid].z;

       float dx1 = coorx - atominfo[atomid].x;
       float dx2 = dx1 + gridspacing_u;
       float dx3 = dx2 + gridspacing_u;
       float dx4 = dx3 + gridspacing_u;
       float dx5 = dx4 + gridspacing_u;
       float dx6 = dx5 + gridspacing_u;
       float dx7 = dx6 + gridspacing_u;
       float dx8 = dx7 + gridspacing_u;

       energyvalx1 += atominfo[atomid].w * (1.0f / sqrtf(dx1*dx1 + dyz2));
       energyvalx2 += atominfo[atomid].w * (1.0f / sqrtf(dx2*dx2 + dyz2));
       energyvalx3 += atominfo[atomid].w * (1.0f / sqrtf(dx3*dx3 + dyz2));
       energyvalx4 += atominfo[atomid].w * (1.0f / sqrtf(dx4*dx4 + dyz2));
       energyvalx5 += atominfo[atomid].w * (1.0f / sqrtf(dx5*dx5 + dyz2));
       energyvalx6 += atominfo[atomid].w * (1.0f / sqrtf(dx6*dx6 + dyz2));
       energyvalx7 += atominfo[atomid].w * (1.0f / sqrtf(dx7*dx7 + dyz2));
       energyvalx8 += atominfo[atomid].w * (1.0f / sqrtf(dx8*dx8 + dyz2));
     }

     energygrid[outaddr]   += energyvalx1;
     energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
     energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
     energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
     energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
     energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
     energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
     energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
 }
}

// This function copies atoms from the CPU to the GPU and
// precalculates (z^2) for each atom.

int copyatomstoconstbuf(float *atoms, int count, float zplane) {
  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  float atompre[4*MAXATOMS];
  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);
  CUERR // check and clear any existing errors

  return 0;
}
