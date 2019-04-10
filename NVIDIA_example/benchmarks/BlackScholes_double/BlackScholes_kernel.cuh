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

//__half2_raw one; one.x = 0x3C00; one.y = 0x3C00;


///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline double cndGPU(double d)
{
    const double       A1 = 0.31938153f;
    const double       A2 = -0.356563782f;
    const double       A3 = 1.781477937f;
    const double       A4 = -1.821255978f;
    const double       A5 = 1.330274429f;
    const double RSQRT2PI = 0.39894228040143267793994605993438f;

    double
    K = (1.0f/ (1.0f + 0.2316419f * fabs(d)));

    double
    cnd = RSQRT2PI * exp(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    double &CallResult,
    double &PutResult,
    double S, //Stock price
    double X, //Option strike
    double T, //Option years
    double R, //Riskless rate
    double V  //Volatility rate
)
{
    double sqrtT, expRT;
    double d1, d2, CNDD1, CNDD2;

    sqrtT = 1.0F/ rsqrt(T);
    d1 = (__logf(S / X) + (R + 0.5f * V * V) * T)/ (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = exp(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__launch_bounds__(128)
__global__ void BlackScholesGPU(
    float2 * __restrict d_CallResult,
    float2 * __restrict d_PutResult,
    float2 * __restrict d_StockPrice,
    float2 * __restrict d_OptionStrike,
    float2 * __restrict d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;

     // Calculating 2 options per thread to increase ILP (instruction level parallelism)
    if (opt < (optN / 2))
    {
        double callResult1, callResult2;
        double putResult1, putResult2;
        BlackScholesBodyGPU(
            callResult1,
            putResult1,
            double(d_StockPrice[opt].x),
            double(d_OptionStrike[opt].x),
            double(d_OptionYears[opt].x),
            double(Riskfree),
            double(Volatility)
        );
        BlackScholesBodyGPU(
            callResult2,
            putResult2,
            double(d_StockPrice[opt].y),
            double(d_OptionStrike[opt].y),
            double(d_OptionYears[opt].y),
            double(Riskfree),
            double(Volatility)
        );
        d_CallResult[opt] = make_float2(callResult1, callResult2);
        d_PutResult[opt] = make_float2(putResult1, putResult2);
	 }
}
