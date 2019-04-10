#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#include "kmeans.h"

#define SCATTER

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<float, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<float, 1, cudaReadModeElementType> t_features_flipped;
texture<float, 1, cudaReadModeElementType> t_clusters;


__constant__ float c_clusters[ASSUMED_NR_CLUSTERS*34];		/* constant memory for cluster centers */

/* ----------------- invert_mapping() --------------------- */
/* inverts data array from row-major to column-major.

   [p0,dim0][p0,dim1][p0,dim2] ...
   [p1,dim0][p1,dim1][p1,dim2] ...
   [p2,dim0][p2,dim1][p2,dim2] ...
										to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
*/
__global__ void invert_mapping(float *input,			/* original */
							   float *output,			/* inverted */
							   int npoints,				/* npoints */
							   int nfeatures)			/* nfeatures */
{
	int point_id = threadIdx.x + blockDim.x*blockIdx.x;	/* id of thread */
	int i;

	if(point_id < npoints){
		for(i=0;i<nfeatures;i++)
			output[point_id + npoints*i] = input[point_id*nfeatures + i];
	}
	return;
}
/* ----------------- invert_mapping() end --------------------- */

/* to turn on the GPU delta and center reduction */
//#define GPU_DELTA_REDUCTION
//#define GPU_NEW_CENTER_REDUCTION


/* ----------------- kmeansPoint() --------------------- */
/* find the index of nearest cluster centers and change membership*/
__global__ void
kmeansPoint(float  *features,			/* in: [npoints*nfeatures] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			float  *clusters,
			float  *block_clusters,
			int    *block_deltas,
			int speed
			//,float* distance_vec
			)
{

	// block ID
	const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
#ifdef SCATTER
  if(block_id %10 < speed){
#else
  if(block_id < speed){
#endif
//if (block_id < speed)
//{
	// point/thread ID
	const unsigned int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;

	int  index = -1;

	if (point_id < npoints)
	{
		int i, j;
		//float min_dist = FLT_MAX;
		//half2 dist;													/* distance square between a point to cluster center */
		half min_dist = __float2half_rn(FLT_MAX);

		/* find the cluster center id with min distance to pt */
		for (i=0; i<nclusters/2; i++) {
			int cluster_base_index = 2*i*nfeatures;					/* base index of cluster centers for inverted array */
			int cluster_base_index_next = (2*i+1)*nfeatures;					/* base index of cluster centers for inverted array */

			half2 ans=__float2half2_rn(0.0);												/* Euclidean distance sqaure */

			for (j=0; j < nfeatures; j++)
			{
				int addr = point_id + j*npoints;					/* appropriate index of data point */
				//float diff = (tex1Dfetch(t_features,addr) -
				//			  c_clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
				half2 c_c = __floats2half2_rn(c_clusters[cluster_base_index + j], c_clusters[cluster_base_index_next + j]);
				half2 feature_temp = __float2half2_rn(tex1Dfetch(t_features,addr));
				half2 diff = (feature_temp-c_c) ;
				//ans += (feature_temp-c_c) * (feature_temp-c_c);
				ans += diff*diff;									/* sum of squares */
			}

			//dist = ans;
			//~ distance_vec[point_id*nclusters + nclusters] = ans;
			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			half high_ = __high2half(ans);
			half low_ = __low2half(ans);
			if (high_ < min_dist) {
				min_dist = high_;
				index    = 2*i;
			}
			if (low_ < min_dist){
				min_dist = low_;
				index    = 2*i+1;
			}

		}

		 membership[point_id] = index;
	}

}else{
	//doing float computation
	const unsigned int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;

	int  index = -1;

	if (point_id < npoints)
	{
		int i, j;
		float min_dist = FLT_MAX;
		float dist;													/* distance square between a point to cluster center */

		/* find the cluster center id with min distance to pt */
		for (i=0; i<nclusters; i++) {
			int cluster_base_index = i*nfeatures;					/* base index of cluster centers for inverted array */
			float ans=0.0;												/* Euclidean distance sqaure */

			for (j=0; j < nfeatures; j++)
			{
				int addr = point_id + j*npoints;					/* appropriate index of data point */
				float diff = (tex1Dfetch(t_features,addr) -
							  c_clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
				ans += diff*diff;									/* sum of squares */
			}
			dist = ans;
			//~ distance_vec[point_id*nclusters + nclusters] = ans;
			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;
			}

		}

		 membership[point_id] = index;
}

}

}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
