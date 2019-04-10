//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//	plasmaKernel_gpu_2
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//#include "../../../include/fast_math.cuh"
//#define SLOW_MATH
#ifdef SLOW_MATH
  #include "../../../include/cuda_math.cuh"
#else //my approximate math lib
  #include "../../../include/fast_math.cuh"
#endif

//#define SCATTER

#define SPEED 4000

typedef struct
{
	half v, x, y, z;

} FOUR_VECTOR_HALF;
typedef struct
{
	half2 v, x, y, z;

} FOUR_VECTOR_HALF2;

#define DOT_mixed(A,B) (__half2half2(A.x)*(B.x)+__half2half2(A.y)*(B.y)+__half2half2(A.z)*(B.z))	// STABLE

__global__ void kernel_gpu_cuda(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str* d_box_gpu,
								FOUR_VECTOR* d_rv_gpu,
								fp* d_qv_gpu,
								FOUR_VECTOR* d_fv_gpu)
{

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	THREAD PARAMETERS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	int bx = blockIdx.x;																// get current horizontal block index (0-n)
	int tx = threadIdx.x;															// get current horizontal thread index (0-n)
	// int ax = bx*NUMBER_THREADS+tx;
	// int wbx = bx;
	int wtx = tx;

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	DO FOR THE NUMBER OF BOXES
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  #ifdef SCATTER
    if(bx %100 < SPEED){ //%10 also ok.  this program 8000 blocks so %100 is good
  #else
    if(bx < SPEED){
  #endif
//	if (bx < SPEED){

	if(bx<d_dim_gpu.number_boxes){
	// while(wbx<box_indexes_counter){

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Extract input parameters
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// parameters
		half2 a2 = __float2half2_rn(2.0*d_par_gpu.alpha*d_par_gpu.alpha);

		// home box
		int first_i;
		FOUR_VECTOR* rA;
		FOUR_VECTOR* fA;
		__shared__ FOUR_VECTOR_HALF rA_shared[100];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR* rB;
		fp* qB;
		int j = 0;
		__shared__ FOUR_VECTOR_HALF2 rB_shared[100/2];
		__shared__ half2 qB_shared[100/2];

		// common
		half2 r2;
		half2 u2;
		half2 vij;
		half2 fs;
		half2 fxij;
		half2 fyij;
		half2 fzij;
		half2_3 d;

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Home box
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Copy to shared memory
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX){
			rA_shared[wtx].x = __float2half_rn(rA[wtx].x);
			rA_shared[wtx].y = __float2half_rn(rA[wtx].y);
			rA_shared[wtx].z = __float2half_rn(rA[wtx].z);
			rA_shared[wtx].v = __float2half_rn(rA[wtx].v);
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++){

			//----------------------------------------50
			//	nei box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = bx;													// set first box to be processed to home box
			}
			else{
				pointer = d_box_gpu[bx].nei[k-1].number;							// remaining boxes are nei boxes
			}

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// nei box - shared memory
			wtx = wtx/2;
			while(wtx<NUMBER_PAR_PER_BOX/2){
				rB_shared[wtx].x = __floats2half2_rn(rB[2*wtx].x, rB[2*wtx+1].x);
				rB_shared[wtx].y = __floats2half2_rn(rB[2*wtx].y, rB[2*wtx+1].y);
				rB_shared[wtx].z = __floats2half2_rn(rB[2*wtx].z, rB[2*wtx+1].z);
				rB_shared[wtx].v = __floats2half2_rn(rB[2*wtx].v, rB[2*wtx+1].v);
				qB_shared[wtx] = __floats2half2_rn(qB[2*wtx], qB[2*wtx+1]);
				wtx = wtx + NUMBER_THREADS/2;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation
			//----------------------------------------------------------------------------------------------------------------------------------140
			FOUR_VECTOR_HALF2 fA_tmp;
			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while(wtx<NUMBER_PAR_PER_BOX){
				fA_tmp.v = __float2half2_rn(0.0);
				fA_tmp.x = __float2half2_rn(0.0);
				fA_tmp.y = __float2half2_rn(0.0);
				fA_tmp.z = __float2half2_rn(0.0);
				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX/2; j++){

					r2 = __half2half2(rA_shared[wtx].v) + rB_shared[j].v - DOT_mixed(rA_shared[wtx],rB_shared[j]);
					u2 = a2*r2;
					vij= fast_h2exp(-u2);
					fs = 2*vij;

					d.x = __half2half2(rA_shared[wtx].x)  - rB_shared[j].x;
					fxij=fs*d.x;
					d.y = __half2half2(rA_shared[wtx].y)  - rB_shared[j].y;
					fyij=fs*d.y;
					d.z = __half2half2(rA_shared[wtx].z)  - rB_shared[j].z;
					fzij=fs*d.z;

					fA_tmp.v += (qB_shared[j]*vij);
					fA_tmp.x += (qB_shared[j]*fxij);
					fA_tmp.y += (qB_shared[j]*fyij);
					fA_tmp.z += (qB_shared[j]*fzij);


				}
				float2 tmp_fa = __half22float2(fA_tmp.v);
				fA[wtx].v += tmp_fa.x + tmp_fa.y;
				tmp_fa = __half22float2(fA_tmp.x);
				fA[wtx].x += tmp_fa.x + tmp_fa.y;
				tmp_fa = __half22float2(fA_tmp.y);
				fA[wtx].y += tmp_fa.x + tmp_fa.y;
				tmp_fa = __half22float2(fA_tmp.z);
				fA[wtx].z += tmp_fa.x + tmp_fa.y;

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;

			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140

		}

		// // increment work block index
		// wbx = wbx + NUMBER_BLOCKS;

		// // synchronize - because next iteration will overwrite current shared memory
		// __syncthreads();

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

	}
} else
{
//  doing float computing
if(bx<d_dim_gpu.number_boxes){
// while(wbx<box_indexes_counter){

	//------------------------------------------------------------------------------------------------------------------------------------------------------160
	//	Extract input parameters
	//------------------------------------------------------------------------------------------------------------------------------------------------------160

	// parameters
	fp a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

	// home box
	int first_i;
	FOUR_VECTOR* rA;
	FOUR_VECTOR* fA;
	__shared__ FOUR_VECTOR rA_shared[100];

	// nei box
	int pointer;
	int k = 0;
	int first_j;
	FOUR_VECTOR* rB;
	fp* qB;
	int j = 0;
	__shared__ FOUR_VECTOR rB_shared[100];
	__shared__ double qB_shared[100];

	// common
	fp r2;
	fp u2;
	fp vij;
	fp fs;
	fp fxij;
	fp fyij;
	fp fzij;
	THREE_VECTOR d;

	//------------------------------------------------------------------------------------------------------------------------------------------------------160
	//	Home box
	//------------------------------------------------------------------------------------------------------------------------------------------------------160

	//----------------------------------------------------------------------------------------------------------------------------------140
	//	Setup parameters
	//----------------------------------------------------------------------------------------------------------------------------------140

	// home box - box parameters
	first_i = d_box_gpu[bx].offset;
	rA = &d_rv_gpu[first_i];
	fA = &d_fv_gpu[first_i];

	//----------------------------------------------------------------------------------------------------------------------------------140
	//	Copy to shared memory
	//----------------------------------------------------------------------------------------------------------------------------------140

	// home box - shared memory
	while(wtx<NUMBER_PAR_PER_BOX){
		rA_shared[wtx] = rA[wtx];
		wtx = wtx + NUMBER_THREADS;
	}
	wtx = tx;

	// synchronize threads  - not needed, but just to be safe
	__syncthreads();

	//------------------------------------------------------------------------------------------------------------------------------------------------------160
	//	nei box loop
	//------------------------------------------------------------------------------------------------------------------------------------------------------160

	// loop over neiing boxes of home box
	for (k=0; k<(1+d_box_gpu[bx].nn); k++){

		//----------------------------------------50
		//	nei box - get pointer to the right box
		//----------------------------------------50

		if(k==0){
			pointer = bx;													// set first box to be processed to home box
		}
		else{
			pointer = d_box_gpu[bx].nei[k-1].number;							// remaining boxes are nei boxes
		}

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// nei box - box parameters
		first_j = d_box_gpu[pointer].offset;

		// nei box - distance, (force), charge and (type) parameters
		rB = &d_rv_gpu[first_j];
		qB = &d_qv_gpu[first_j];

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// nei box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX){
			rB_shared[wtx] = rB[wtx];
			qB_shared[wtx] = qB[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads because in next section each thread accesses data brought in by different threads here
		__syncthreads();

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Calculation
		//----------------------------------------------------------------------------------------------------------------------------------140

		// loop for the number of particles in the home box
		// for (int i=0; i<nTotal_i; i++){
		while(wtx<NUMBER_PAR_PER_BOX){

			// loop for the number of particles in the current nei box
			for (j=0; j<NUMBER_PAR_PER_BOX; j++){

				// r2 = rA[wtx].v + rB[j].v - DOT(rA[wtx],rB[j]);
				// u2 = a2*r2;
				// vij= exp(-u2);
				// fs = 2.*vij;

				// d.x = rA[wtx].x  - rB[j].x;
				// fxij=fs*d.x;
				// d.y = rA[wtx].y  - rB[j].y;
				// fyij=fs*d.y;
				// d.z = rA[wtx].z  - rB[j].z;
				// fzij=fs*d.z;

				// fA[wtx].v +=  qB[j]*vij;
				// fA[wtx].x +=  qB[j]*fxij;
				// fA[wtx].y +=  qB[j]*fyij;
				// fA[wtx].z +=  qB[j]*fzij;



				r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[j].v - DOT((fp)rA_shared[wtx],(fp)rB_shared[j]);
				u2 = a2*r2;
				vij= exp(-u2);
				fs = 2*vij;

				d.x = (fp)rA_shared[wtx].x  - (fp)rB_shared[j].x;
				fxij=fs*d.x;
				d.y = (fp)rA_shared[wtx].y  - (fp)rB_shared[j].y;
				fyij=fs*d.y;
				d.z = (fp)rA_shared[wtx].z  - (fp)rB_shared[j].z;
				fzij=fs*d.z;

				fA[wtx].v += (double)((fp)qB_shared[j]*vij);
				fA[wtx].x += (double)((fp)qB_shared[j]*fxij);
				fA[wtx].y += (double)((fp)qB_shared[j]*fyij);
				fA[wtx].z += (double)((fp)qB_shared[j]*fzij);

			}

			// increment work thread index
			wtx = wtx + NUMBER_THREADS;

		}

		// reset work index
		wtx = tx;

		// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
		__syncthreads();

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Calculation END
		//----------------------------------------------------------------------------------------------------------------------------------140

	}

	// // increment work block index
	// wbx = wbx + NUMBER_BLOCKS;

	// // synchronize - because next iteration will overwrite current shared memory
	// __syncthreads();

	//------------------------------------------------------------------------------------------------------------------------------------------------------160
	//	nei box loop END
	//------------------------------------------------------------------------------------------------------------------------------------------------------160

}




} //end float kernel



}
