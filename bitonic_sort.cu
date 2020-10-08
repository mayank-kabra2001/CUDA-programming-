#include<cuda.h>
#include<cuda_runtime.h>

// ###################### CPU PROGRAM ################################

// void compare(int i , int l , bool dir)
// {
// 	if(dir == (a[i] > a[j]))
// 	{
// 		exchange(i , j) ; 
// 	}

// }

// void bitonicMerge(int lo , int n , bool dir) 
// {
// 	if(n>1)
// 	{
// 		int m = n/2 ; 
// 		for(int i=lo ; i<lo+m ; i++)
// 		{
// 			compare(i , i+n , dir) ;
// 		}
// 		bitonicMerge(lo , m , dir); 
// 		bitonicMerge(lo+m , m , dir) ; 
// 	}
// }

// void bitonicSort(int lo , int n , bool dir)
// {
// 	if(n>1) 
// 	{
// 		int m =n/2 ; 
// 		bitonicSort(lo , m , ASCENDING) ; 
// 		bitonicSort(lo+n , m , DESCENDING) ; 
// 		bitonicSort(lo , n , dir) ; 
// 	}
// }

################# GPU PROGRAM ######################################

__device__ inline void Comparator(uint &keyA , uint &valA , uint &keyB , uint &valB , uint dir) 
{
	uint t ; 
	if((keyA > keyB) == dir)
	{
		t = keyA ; 
		keyA = keyB ; 
		keyB = t ; 
		t = valA ; 
		valA = valB ; 
		valB = t ; 
	}
}

__global__ void bitonicSortShared1(uint *d_Dstkey , uint *d_Dstval , uint *d_Srckey , uint *d_Srcval)
{
	__shared__ uint s_key[SHARED_SIZE] ; 
	__shared__ uint s_val[SHARED_SIZE] ; 

	d_Srckey += blockIdx.x * SHARED_SIZE + threadIdx.x ; 
	d_Srcval += blockIdx.x * SHARED_SIZE + threadIdx.x ; 
	d_Dstkey += blockIdx.x * SHARED_SIZE + threadIdx.x ; 
	d_Dstval += blockIdx.x * SHARED_SIZE + threadIdx.x ; 
	

}