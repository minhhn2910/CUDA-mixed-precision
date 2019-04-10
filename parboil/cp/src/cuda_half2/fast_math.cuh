#ifndef FAST_MATH_CUH
#define FAST_MATH_CUH
#include <cuda_fp16.h>
//important macroes in cuda_fp16.hpp for corectness (without my messy type & pointer casting
/*
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __HALF_TO_VUS(var) *(reinterpret_cast<volatile unsigned short *>(&(var)))
#define __HALF_TO_CVUS(var) *(reinterpret_cast<const volatile unsigned short *>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))
*/
//79 cycles half2 division on Nvidia V100
__device__ half2 fast_h2rcp(half2 input){
  half2 c0 = __float2half2_rn(2.823529);
  half2 c1 = __float2half2_rn(-1.8823529);
  half2 four = __float2half2_rn(4.0);
  half2 one = __float2half2_rn(1.0);

  half2 temp_2 = input*four;
  int c = *((int*) (&temp_2));
  c= (c ^ 0x7c007c00 ) & 0xfc00fc00 ;
  half2 factor = (*(half2*)(&c));

  half2 d = input * factor;

  half2 x =  d*c1;
  x.x = __hadd(x.x, c0.x);
  x.y = __hadd(x.y, c0.y);
  x = x + x * one - x * x * d;

  return x*factor;
}

__device__ half2 fast_h2exp(half2 input){

    half2 result = __float2half2_rn(1477)*input;
   result.x += __float2half_rn(15293.36);
   result.y += __float2half_rn(15293.36);
   short2 result_short2;
    result_short2.x = (short)(result.x);
    result_short2.y = (short)(result.y);
    return *(half2*)(&result_short2);
}

__device__ half2 exp_half2_saturated(half2 input) {
     half2 result = __float2half2_rn(1477)*input;
    result.x += __float2half_rn(15293.36);
   result.y += __float2half_rn(15293.36);
   short2 result_short2;
    result_short2.x = (short)(result.x);
    result_short2.y = (short)(result.y);

    if(input.x < __float2half_rn(-10))
        result_short2.x  = 0;
    if(input.x > __float2half_rn(10))
        //31743 = 7BFF (65504 - largest normal number)
        result_short2.x  = 31743;
    if(input.y < __float2half_rn(-10))
        result_short2.y = 0;
    if(input.y > __float2half_rn(10))
        result_short2.x  = 31743;

  return *(half2*)(&result_short2);
}

//100 cycles
__device__ half2 fast_h2log2(half2 input){
    uint32_t input_int = *(uint32_t*)&input;
    uint32_t exp_raw = (input_int>>10) & 0x001f001f; //get 5 bit exponent only
    half2 exp_raw_half2 = *(half2*)&exp_raw;
    short exp1 = __half_as_ushort(exp_raw_half2.x) - 15;
    short exp2 = __half_as_ushort(exp_raw_half2.y) - 15;
    __half2_raw exp_converted ;
    exp_converted.x = __short2half_rn(exp1); 
    exp_converted.y = __short2half_rn(exp2);
    uint32_t mantissa = input_int & 0x03ff03ff;
    uint32_t normalized_int = mantissa | 0x3c003c00;
    half2 normalized = *(half2*)&normalized_int;
    return exp_converted  + normalized - 1.0 + 0.045;
}

//50 cycles, no cvt (type conversion instructions)
__device__  half2 fastest_h2log2(half2 x){
  // exp range from -15 -> 15;
  //uint32_t x_half2 = floats2half2(x,x*4); //e and e+2
  uint32_t x_half2 = *(reinterpret_cast<const unsigned int *>(&(x)));
  uint32_t exp_raw = (x_half2>>10) & 0x001f001f;

  //uint32_t magic_num = __float2half2_rn(32.0);
  uint32_t  magic_num = 0x50005000 | exp_raw;
  half2 exp_converted = *(half2*)&magic_num;
  exp_converted = (exp_converted - 32.0)*32.0 -15.0;
  uint32_t normalized_int = (x_half2 & 0x03ff03ff) | 0x3c003c00;
  half2 normalized = *(half2*)&normalized_int;
  return exp_converted  + normalized - __float2half2_rn(0.955);
}

__device__ half2 fast_h2log(half2 input){
    //0.6931 = ln(2)
    return fastest_h2log2(input)*0.6931;
}

__device__ half2 fast_h2log10(half2 input){

    //0.301 = log10(2);
    return fastest_h2log2(input)*0.301;
}


// magic number 0x1DE9
__device__ half2 slow_h2sqrt(half2 input){

  uint32_t nosign  = (*(uint32_t*)(&input)) & 0x7fff7fff;
//  float xhalf = half(0.5f * abs_x);
  half2 abs_x = *((half2*)&nosign);
  half2 xhalf = __float2half2_rn(0.5f) * abs_x;
  uint32_t sign_only = 0x80008000 &  (*((uint32_t*)&input));
 // uint32_t nosign = *((uint32_t*)&abs_x);

  nosign = 0x1DE91DE9 - (nosign >> 1);  //
  nosign = nosign & 0x7fff7fff;
  half2 nosign_half2 = *((half2*)& nosign);
//  half2 result = nosign_half2*(__float2half2_rn(1.5f)-(xhalf*nosign_half2*nosign_half2));
  half2 result = nosign_half2*__float2half2_rn(1.5f);

  result -=xhalf*nosign_half2*nosign_half2*nosign_half2;

  return result;

}

//54 cycles half2 rsqrt trick on Nvidia V100
__device__ half2 fast_h2rsqrt( half2 input) {

  uint32_t nosign  = (*(uint32_t*)(&input)) & 0x7fff7fff;
//  float xhalf = half(0.5f * abs_x);
  half2 abs_x = *((half2*)&nosign);
  half2 xhalf = __float2half2_rn(0.5f) * abs_x;
  uint32_t sign_only = 0x80008000 &  (*((uint32_t*)&input));
  //sign can multiply;
  //3c00 = 1 ; Bc00 = -1;
  uint32_t multiply_fact = 0x3c003c00 ^ sign_only;

 // uint32_t nosign = *((uint32_t*)&abs_x);

  nosign = 0x59BB59BB - (nosign >> 1);  //
  nosign = nosign & 0x7fff7fff;
  half2 nosign_half2 = *((half2*)& nosign);
//  half2 result = nosign_half2*(__float2half2_rn(1.5f)-(xhalf*nosign_half2*nosign_half2));
  half2 result = nosign_half2*__float2half2_rn(1.5f);

  result -=xhalf*nosign_half2*nosign_half2*nosign_half2;

  return result;
}

__device__ half2 fast_h2sin(half2 input){

    return input;
}

__device__ half2 fast_h2cos(half2 input){
    return input;
}

__device__ half2 fast_h2asin(half2 x){
    
    half2 coeff_3 = __float2half2_rn(0.16667);
    half2 coeff_5 = __float2half2_rn(0.075);
    half2 coeff_7 = __float2half2_rn(0.04464);
    half2 x_2 = x*x;
    half2 x_3 = x_2*x;
    return x + coeff_3*x_3 + coeff_5*x_2*x_3 + coeff_7*x_3*x_3*x;
}

__device__ half2 fast_h2acos(half2 input){

    return __float2half2_rn(1.5708) - fast_h2asin(input);;
}

double RandNum (double min, double max)
{
   return min + (max - min) * ((double)rand()
                            / (double)RAND_MAX);
}


/*
float fastlog2(float input){
  //assume input > 0;
  int input_int = *(int*)&input;
  int exp = (input_int>>23)-127;
  //3f80 0000 1*2^0;
  //007f ffff mantissa only
  int mantissa = input_int & 0x007fffff;
  int normalized_int = mantissa | 0x3f800000;
  float normalized = *(float*)&(normalized_int);

  //half 
  short exp = (input_short>>10)-15;
  //3c00 1*2^0;
  //03ff mantissa only
  short mantissa = input_short & 0x03ff;
  short normalized_short = mantissa | 0x3c00;

  printf("normalized %f  exp : %d ", normalized, exp);
  //return exp * 0.30 ; //wrong +-0.3;
  float result = exp  + normalized - 1 + 0.045;
  return result;
}
float fastlog10(float input ){
  //0.301 = log10(2);
  return fastlog2(input)*0.301;
}
float fastlog(float input ){
  //0.6931 = ln(2)
  return fastlog2(input)*0.6931;
}
*/
#endif
