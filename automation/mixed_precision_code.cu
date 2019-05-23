#include "stdlib.h"
#include <fstream>
#include <iostream>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define MAX_LOOP 1000
#define MAX_DIFF 0.15f
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)

using namespace std;

#ifdef SLOW_MATH
#include "../include/cuda_math.cuh"
#else
#include "../include/fast_math.cuh"
#endif

#define SCATTER

using namespace std;
__global__ void invkin_kernel(float *xTarget_in, float *yTarget_in, float *angles, int size, float err_thresh, int speed) {
#ifdef SCATTER
	if(blockIdx.x %100 < speed) {
#else
	if(blockIdx.x < speed) {
#endif

	if(threadIdx.x<blockDim.x/2){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + 2*threadIdx.x;
	if(idx < size)
	{
    	half2 angle_out[NUM_JOINTS];

			half zero = 0.0f;
			half one = 1.f;
			half minus_one = -1.f;

    	for(int i = 0; i < NUM_JOINTS; i++)
    	{
  			angle_out[i] = __float2half2_rn(0.0);
				//angle_out[i].x = 0.f;
				//angle_out[i].y = 0.f;
    	}

    	half max_err 	= err_thresh * (float)(NUM_JOINTS);
    	half err 		= max_err + one; // initialize error to something greater than error threshold


		// Initialize x and y data
		half2 xData[NUM_JOINTS_P1];
		half2 yData[NUM_JOINTS_P1];

		for (int i = 0 ; i < NUM_JOINTS_P1; i++)
		{
			xData[i] = __float2half2_rn((float)i);
			yData[i] = __float2half2_rn(0.f);
		}

		half2 xTarget_in_temp = __floats2half2_rn(xTarget_in[idx],xTarget_in[idx+1]);
		half2 yTarget_in_temp = __floats2half2_rn(yTarget_in[idx],yTarget_in[idx+1]);


		//half minus_one = -1.0f;

		half2 pe_x = xData[NUM_JOINTS];
		half2 pe_y = yData[NUM_JOINTS];

		for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
		{
			for (int iter = NUM_JOINTS; iter > 0; iter--)
			{
				half2 pc_x = xData[iter-1];
				half2 pc_y = yData[iter-1];
				half2 diff_pe_pc_x = pe_x - pc_x;
				half2 diff_pe_pc_y = pe_y - pc_y;
//				half2 diff_tgt_pc_x = xTarget_in[idx] - pc_x;
//				half2 diff_tgt_pc_y = yTarget_in[idx] - pc_y;
				half2 diff_tgt_pc_x = xTarget_in_temp - pc_x;
				half2 diff_tgt_pc_y = yTarget_in_temp - pc_y;
				half2 len_diff_pe_pc = fast_h2sqrt (diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
				half2 len_diff_tgt_pc = fast_h2sqrt (diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
				half2 a_x = diff_pe_pc_x * fast_h2rcp(len_diff_pe_pc);
				half2 a_y = diff_pe_pc_y * fast_h2rcp(len_diff_pe_pc);
				half2 b_x = diff_tgt_pc_x * fast_h2rcp(len_diff_tgt_pc);
				half2 b_y = diff_tgt_pc_y * fast_h2rcp(len_diff_tgt_pc);
				half2 a_dot_b = a_x * b_x + a_y * b_y;


				//float2 a_dot_b_float  = __half22float2(a_dot_b);

				if (a_dot_b.x > one) {
					a_dot_b.x = one ;
				}
				if (a_dot_b.x < minus_one) {
					a_dot_b.x = minus_one ;
				}
				if (a_dot_b.y > one) {
					a_dot_b.y = one ;
				}
				if (a_dot_b.y < minus_one) {
					a_dot_b.y = minus_one ;
				}

		/*
				if (a_dot_b > 1.f)
					a_dot_b = 1.f;
				else if (a_dot_b < -1.f)
					a_dot_b = -1.f;
*/


				//float2 a_dot_b_float = __half22float2(a_dot_b);
				//half2 angle =__floats2half2_rn (acosf(a_dot_b_float.x) * (180.f / PI), acosf(a_dot_b_float.x) * (180.f / PI));
				//angle.x = acosf(a_dot_b_float.x) * (180.f / PI);
				//angle.y = acosf(a_dot_b_float.y) * (180.f / PI);
				half2 angle = fast_h2acos(a_dot_b) * 57.29578;//(180.f / PI);
				// Determine angle direction

				half2 direction = a_x * b_y - a_y * b_x;
				if (direction.x < zero)

					angle.x = -angle.x ;
				if (direction.y < zero)
					angle.y = -angle.y;

				// Make the result look more natural (these checks may be omitted)
				// if (angle > 30.f)
				// 	angle = 30.f;
				// else if (angle < -30.f)
				// 	angle = -30.f;
				// Save angle
				angle_out[iter - 1] = angle;
				for (int i = 0; i < NUM_JOINTS; i++)
				{
					if(i < NUM_JOINTS - 1)
					{
						angle_out[i+1] += angle_out[i];
						//angle_out[i+1].y += angle_out[i].y;
					}
				}
			}// loop NUM_JOINTS
		}// loop 1k


		float2 angle_0 = __half22float2(angle_out[0]);
		float2 angle_1 = __half22float2(angle_out[1]);
		float2 angle_2 = __half22float2(angle_out[2]);

		angles[idx * NUM_JOINTS + 0] = angle_0.x;
		angles[idx * NUM_JOINTS + 1] = angle_1.x;
		angles[idx * NUM_JOINTS + 2] = angle_2.x;

		angles[(idx+1) * NUM_JOINTS + 0] = angle_0.y;
		angles[(idx+1) * NUM_JOINTS + 1] = angle_1.y;
		angles[(idx+1) * NUM_JOINTS + 2] = angle_2.y;
	}


} //end if(threadIdx.x<512/2)
else return;
} else {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if(idx < size)
	{
//		float parrotInput[2];
//    	float parrotOutput[3];

    	float angle_out[NUM_JOINTS];

    	for(int i = 0; i < NUM_JOINTS; i++)
    	{
  			angle_out[i] = 0.0;
    	}

    	float max_err 	= err_thresh * (float)(NUM_JOINTS);
    	float err 		= max_err + 1.f; // initialize error to something greater than error threshold

/* parrot not used
    	parrotInput[0] = xTarget_in[idx];
    	parrotInput[1] = yTarget_in[idx];

#pragma parrot(input, "invkin_kernel", [2]<-1.0; 1.0>parrotInput)
*/
		//float max_err = err_thresh * (float)(NUM_JOINTS);
		//float err = max_err + 1.f;

		// Initialize x and y data
		float xData[NUM_JOINTS_P1];
		float yData[NUM_JOINTS_P1];

		for (int i = 0 ; i < NUM_JOINTS_P1; i++)
		{
			xData[i] = i;
			yData[i] = 0.f;
		}

		for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
		{
			for (int iter = NUM_JOINTS; iter > 0; iter--)
			{
				float pe_x = xData[NUM_JOINTS];
				float pe_y = yData[NUM_JOINTS];
				float pc_x = xData[iter-1];
				float pc_y = yData[iter-1];
				float diff_pe_pc_x = pe_x - pc_x;
				float diff_pe_pc_y = pe_y - pc_y;
				float diff_tgt_pc_x = xTarget_in[idx] - pc_x;
				float diff_tgt_pc_y = yTarget_in[idx] - pc_y;
				float len_diff_pe_pc = sqrt(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
				float len_diff_tgt_pc = sqrt(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
				float a_x = diff_pe_pc_x / len_diff_pe_pc;
				float a_y = diff_pe_pc_y / len_diff_pe_pc;
				float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
				float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
				float a_dot_b = a_x * b_x + a_y * b_y;
				if (a_dot_b > 1.f)
					a_dot_b = 1.f;
				else if (a_dot_b < -1.f)
					a_dot_b = -1.f;
				float angle = acos(a_dot_b) * (180.f / PI);
				// Determine angle direction
				float direction = a_x * b_y - a_y * b_x;
				if (direction < 0.f)
					angle = -angle;
				// Make the result look more natural (these checks may be omitted)
				// if (angle > 30.f)
				// 	angle = 30.f;
				// else if (angle < -30.f)
				// 	angle = -30.f;
				// Save angle
				angle_out[iter - 1] = angle;
				for (int i = 0; i < NUM_JOINTS; i++)
				{
					if(i < NUM_JOINTS - 1)
					{
						angle_out[i+1] += angle_out[i];
					}
				}
			}
		}

/*		parrot : not used
		parrotOutput[0] = angle_out[0] / 30.0;
		parrotOutput[1] = angle_out[1] / 30.0;
		parrotOutput[2] = angle_out[2] / 30.0;

#pragma parrot(output, "invkin_kernel", [3]<-1.0; 1.0>parrotOutput)

		angle_out[0] = parrotOutput[0] * 30.0;
		angle_out[1] = parrotOutput[1] * 30.0;
		angle_out[2] = parrotOutput[2] * 30.0;
*/
		angles[idx * NUM_JOINTS + 0] = angle_out[0];
		angles[idx * NUM_JOINTS + 1] = angle_out[1];
		angles[idx * NUM_JOINTS + 2] = angle_out[2];
	}
}
}

int main(int argc, char* argv[])
{
	int speed = 50;
	std::cout << "# Speed = " << speed << std::endl;

	if(argc != 4)
	{
		std::cerr << "Usage: ./invkin.out <input file coefficients> <output file> <error threshold>" << std::endl;
		exit(EXIT_FAILURE);
	}

	float* xTarget_in_h;
	float* yTarget_in_h;
	float* angle_out_h;

	cudaError_t cudaStatus;

	int data_size = 0;

	// process the files
	ifstream coordinate_in_file (argv[1]);
	ofstream angle_out_file (argv[2]);
	float err_thresh = atof(argv[3]);


	if(coordinate_in_file.is_open())
	{
		coordinate_in_file >> data_size;
		std::cout << "# Data Size = " << data_size << std::endl;
	}

	// allocate the memory
	xTarget_in_h = new (nothrow) float[data_size];
	if(xTarget_in_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	yTarget_in_h = new (nothrow) float[data_size];
	if(yTarget_in_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	angle_out_h = new (nothrow) float[data_size*NUM_JOINTS];
	if(angle_out_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}


	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// add data to the arrays
	float xTarget_tmp, yTarget_tmp;
	int coeff_index = 0;
	while(coeff_index < data_size)
	{
		coordinate_in_file >> xTarget_tmp >> yTarget_tmp;

		for(int i = 0; i < NUM_JOINTS ; i++)
		{
			angle_out_h[coeff_index * NUM_JOINTS + i] = 0.0;
		}

		xTarget_in_h[coeff_index] = xTarget_tmp;
		yTarget_in_h[coeff_index++] = yTarget_tmp;
	}


	std::cout << "# Coordinates are read from file..." << std::endl;

	// memory allocations on the host
	float 	*xTarget_in_d,
			*yTarget_in_d;
	float 	*angle_out_d;

	cudaMalloc((void**) &xTarget_in_d, data_size * sizeof(float));
	cudaMalloc((void**) &yTarget_in_d, data_size * sizeof(float));
	cudaMalloc((void**) &angle_out_d,  data_size * NUM_JOINTS * sizeof(float));

	std::cout << "# Memory allocation on GPU is done..." << std::endl;

	cudaMemcpy(xTarget_in_d, xTarget_in_h, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yTarget_in_d, yTarget_in_h, data_size * sizeof(float), cudaMemcpyHostToDevice);

	std::cout << "# Data are transfered to GPU..." << std::endl;

	dim3 dimBlock	( 512, 1 );
	dim3 dimGrid	( data_size / 512, 1 );


	cudaEventRecord(start, 0);

#pragma parrot.start("invkin_kernel")

	invkin_kernel<<<dimGrid, dimBlock>>>(xTarget_in_d, yTarget_in_d, angle_out_d, data_size, err_thresh, speed);

#pragma parrot.end("invkin_kernel")

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
       	std::cout << "Something was wrong! Error code: " << cudaStatus << std::endl;
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << "# Elapsed Time in `nrpoly3` kernel = " << elapsedTime << std::endl;
	std::cout << "# GPU computation is done ..." << std::endl;

	cudaMemcpy(angle_out_h, angle_out_d, data_size * NUM_JOINTS * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < data_size; i++)
	{
		// angle_out_file << xTarget_in_h[i] << " " << yTarget_in_h[i] << " ";
		//compare output, not need to store this
		for(int j = 0 ; j < NUM_JOINTS; j++)
		{
			angle_out_file << angle_out_h[i * NUM_JOINTS + j] << " ";
		}
		angle_out_file << std::endl;
	}

	// close files
	coordinate_in_file.close();
	angle_out_file.close();

	// de-allocate the memory
	delete[] xTarget_in_h;
	delete[] yTarget_in_h;
	delete[] angle_out_h;

	// de-allocate cuda memory
	cudaFree(xTarget_in_d);
	cudaFree(yTarget_in_d);
	cudaFree(angle_out_d);

	std::cout << "Thank you..." << std::endl;
}