// Designed by: Amir Yazdanbakhsh
// Date: March 26th - 2015
// Alternative Computing Technologies Lab.
// Georgia Institute of Technology


#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <cstddef>

// Cuda Libraries
#include <cuda_runtime_api.h>
#include <cuda.h>

//#define MAX_LOOP 5
#define MAX_LOOP 20 //more loops, even number

#define MAX_DIFF 0.15f

#define F(x) 	(A_coeff[idx]*x*x*x)+(B_coeff[idx]*x*x)+(C_coeff[idx]*x)+D_coeff[idx]
#define FD(x) 	(3.0*A_coeff[idx]*x*x)+(2.0*B_coeff[idx]*x)+(C_coeff[idx])

using namespace std;

__global__ void nrpol3_kernel(float *A_coeff, float *B_coeff, float *C_coeff, float *D_coeff, float *x0_in, float *root, int size, float err_thresh)
{

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if(idx < size)
	{

/*		float parrotInput[5];
    	float parrotOutput[1];

    	parrotInput[0] = A_coeff[idx] / 20.0;
    	parrotInput[1] = B_coeff[idx] / 20.0;
    	parrotInput[2] = C_coeff[idx] / 20.0;
    	parrotInput[3] = D_coeff[idx] / 4000.0;
    	parrotInput[4] = x0_in[idx]   / 15000.0;
			parrot is not used, test on real device;

#pragma parrot(input, "nrpol3_kernel", [5]parrotInput)
*/
		float curr_err = MAX_DIFF * err_thresh;
		float x0 = x0_in[idx];
		float x1, fx, fdx;
		float temp_result = 0;//store temp_res in loop
		for (int i = 0; i < MAX_LOOP; i++) {
			//fx  = F(x0);
			fx = (A_coeff[idx]*x0*x0*x0)+(B_coeff[idx]*x0*x0)+(C_coeff[idx]*x0)+D_coeff[idx];
			//fdx = FD(x0);
			fdx = (3.0*A_coeff[idx]*x0*x0)+(2.0*B_coeff[idx]*x0)+(C_coeff[idx]);
			x1  = x0 - (fx/fdx);
			if (fabs((x1-x0) / x1) < curr_err) {
				curr_err = fabs((x1-x0) / x1);
				temp_result = x1;
			}
			x0 = x1;
		}
//		parrotOutput[0] = root[idx] / 20.0;
//#pragma parrot(output, "nrpol3_kernel", [1]<-2.0; 2.0>parrotOutput)
		root[idx] = temp_result;//parrotOutput[0] * 20.0;

	}
}

int main(int argc, char* argv[])
{
	if(argc != 4)
	{
		std::cerr << "Usage: ./nrpoly3.out <input file coefficients> <output file> <error threshold>" << std::endl;
		exit(EXIT_FAILURE);
	}

	float* A_coeff;
	float* B_coeff;
	float* C_coeff;
	float* D_coeff;
	float* x0;
	float* root;

	cudaError_t cudaStatus;

	int data_size = 0;

	// process the files
	ifstream coeff_in_file (argv[1]);
	ofstream root_out_file (argv[2]);
	float err_thresh = atof(argv[3]);


	if(coeff_in_file.is_open())
	{
		coeff_in_file >> data_size;
		std::cout << "# Data Size = " << data_size << std::endl;
	}

	// allocate the memory
	A_coeff = new (nothrow) float[data_size];
	if(A_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	B_coeff = new (nothrow) float[data_size];
	if(B_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	C_coeff = new (nothrow) float[data_size];
	if(C_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	D_coeff = new (nothrow) float[data_size];
	if(D_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	x0 = new (nothrow) float[data_size];
	if(x0 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	root = new (nothrow) float[data_size];
	if(root == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}


	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// add data to the arrays
	float A_tmp, B_tmp, C_tmp, D_tmp, x0_tmp;
	int coeff_index = 0;
	while(coeff_index < data_size)
	{
		coeff_in_file >> A_tmp >> B_tmp >> C_tmp >> D_tmp >> x0_tmp;

		root 	[coeff_index]   = 0;
		A_coeff	[coeff_index] 	= A_tmp;
		B_coeff	[coeff_index] 	= B_tmp;
		C_coeff	[coeff_index]	= C_tmp;
		D_coeff	[coeff_index] 	= D_tmp;
		x0 		[coeff_index++] = x0_tmp;
	}


	std::cout << "# Coefficients are read from file..." << std::endl;

	// memory allocations on the host
	float 	*A_coeff_d,
			*B_coeff_d,
			*C_coeff_d,
			*D_coeff_d,
			*x0_d;
	float 	* root_d;

	cudaMalloc((void**) &A_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &B_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &C_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &D_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &x0_d, 		data_size * sizeof(float));
	cudaMalloc((void**) &root_d,	data_size * sizeof(float));

	std::cout << "# Memory allocation on GPU is done..." << std::endl;

	cudaMemcpy(A_coeff_d, A_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_coeff_d, B_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_coeff_d, C_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(D_coeff_d, D_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(x0_d, 	  x0,    data_size * sizeof(float), cudaMemcpyHostToDevice);

	std::cout << "# Data are transfered to GPU..." << std::endl;

	dim3 dimBlock	( 512, 1 );
	dim3 dimGrid	( data_size / 512, 1 );


	cudaEventRecord(start, 0);

#pragma parrot.start("nrpol3_kernel")

	nrpol3_kernel<<<dimGrid, dimBlock>>>(A_coeff_d, B_coeff_d, C_coeff_d, D_coeff_d, x0_d, root_d, data_size, err_thresh);

#pragma parrot.end("nrpol3_kernel")

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

	cudaMemcpy( root, root_d, data_size * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < data_size; i++)
	{
		root_out_file << root[i] << std::endl;
	}

	// close files
	root_out_file.close();
	coeff_in_file.close();

	// de-allocate the memory
	delete[] A_coeff;
	delete[] B_coeff;
	delete[] C_coeff;
	delete[] D_coeff;
	delete[] x0;
	delete[] root;

	// de-allocate cuda memory
	cudaFree(A_coeff_d);
	cudaFree(B_coeff_d);
	cudaFree(C_coeff_d);
	cudaFree(D_coeff_d);
	cudaFree(x0_d);
	cudaFree(root_d);

	std::cout << "Thank you..." << std::endl;
}
