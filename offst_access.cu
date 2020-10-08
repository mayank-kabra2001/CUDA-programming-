
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void offset_access(float* a , int s)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x ; 
	a[tid+s] = a[tid + s] + 1; 
}


__global__ void strided_access(float* a , int s)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x ; 
	a[tid*s] = a[tid*s] + 1; 
}

int main()
{
	cudaEvent_t startEvent , stopEvent ; 
	float ms ;
	int blockSize = 1024 ; 
	int n = nMB*1024*1024/sizeof(float) ; 
	cudaMalloc(&d_a , n*sizeof(float)) ; 

	for(int i=0 ; i<32 ; i++)
	{
		cudaMemset(d_A , 0.0 , n*sizeof(float)) ; 
		cudaEventRecord(startEvent) ; 
		offset_access <<< n/blockSize , blockSize >>> (d_a , i) ; 
		cudaEventSynchronize(stopEvent) ;
		cudaEventElapsedTIme(&ms , startEvent , stopEvent) ; 
		printf("%d , %f \n" , i , 2*nMB/ms) ; 
	} 

}