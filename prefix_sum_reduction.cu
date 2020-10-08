#include<cuda.h>
#include<cuda_runtime.h>

__global__ void reduce1(int *g_idata , int *g_odata , unsigned int n)
{
	int *sdata = SharedMemory <int> () ; 

	unsigned int tid = threadIdx.x ; 
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x ; 
	sdata[tid] = (i<n) ? g_idata[i] : 0 ; 
	__syncthreads() ; 

	for(unsigned int s=1 ; s<blockDim.x ; s*=2)
	{
		if((tid%(2*s)) == 0) 
		{
			sdata[tid] += sdata[tid + s] ;  
		}

		__syncthreads() ;
	}

	if(tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0] ; 
	}
}

//
__global__ void reduce2(int *g_idata , int *g_odata , unsigned int n)
{
	int *sdata = SharedMemory <int> () ; 

	unsigned int tid = threadIdx.x ; 
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x ; 
	sdata[tid] = (i<n) ? g_idata[i] : 0 ; 
	__syncthreads() ; 

	for(unsigned int s=1 ; s<blockDim.x ; s*=2)
	{

		//// changed code /////
		int index = 2*s + tid ; 

		if(index < blockDim.x)
		{
			sdata[index] += sdata[index + s] ; 
		}

		/////changed code ///////

		__syncthreads() ;
	}

	if(tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0] ; 
	}
}

// sequential addressing 
__global__ void reduce3(int *g_idata , int *g_odata , unsigned int n)
{
	int *sdata = SharedMemory <int> () ; 

	unsigned int tid = threadIdx.x ; 
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x ; 
	sdata[tid] = (i<n) ? g_idata[i] : 0 ; 
	__syncthreads() ; 

	for(unsigned int s=blockDim.x/2 ; s>0 ; s>>=1)
	{

		//// changed code /////

		if(tid < s)
		{
			sdata[index] += sdata[index + s] ; 
		}

		/////changed code ///////

		__syncthreads() ;
	}

	if(tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0] ; 
	}
}


// first add during load 
__global__ void reduce4(int *g_idata , int *g_odata , unsigned int n)
{
	int *sdata = SharedMemory <int> () ; 

	unsigned int tid = threadIdx.x ; 

	/// changed code ////
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x ; 
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x] ;
	//// changed code ////

	__syncthreads() ; 

	for(unsigned int s=blockDim.x/2 ; s>0 ; s>>=1)
	{

		//// changed code /////

		if(tid < s)
		{
			sdata[index] += sdata[index + s] ; 
		}

		/////changed code ///////

		__syncthreads() ;
	}

	if(tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0] ; 
	}
}

Template <unsigned int blockSize>
__device__ void warpReduce(int* sdata , int tid)
{
	sdata[tid] += sdata[tid + 32] ; 
	sdata[tid] += sdata[tid + 16] ; 
	sdata[tid] += sdata[tid + 8] ; 
	sdata[tid] += sdata[tid + 4] ; 
	sdata[tid] += sdata[tid + 2] ; 
	sdata[tid] += sdata[tid + 1] ; 	
}

/// reducing iterations 
__global__ void reduce5(int *g_idata , int *g_odata , unsigned int n)
{
	int *sdata = SharedMemory <int> () ; 

	unsigned int tid = threadIdx.x ; 

	/// changed code ////
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x ; 
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x] ;
	//// changed code ////

	__syncthreads() ; 

	for(unsigned int s=blockDim.x/2 ; s>32 ; s>>=1)
	{

		//// changed code /////

		if(tid < s)
		{
			sdata[index] += sdata[index + s] ; 
		}

		/////changed code ///////

		__syncthreads() ;
	}

	if(tid < 32) 
	{
		warpReduce(sdata , tid)  
	}

	if(tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0] ; 
	}
}

///complete unrolling
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata , int *g_odata , unsigned int n)
{
	int *sdata = SharedMemory <int> () ; 

	unsigned int tid = threadIdx.x ; 

	/// changed code ////
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x ; 
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x] ;
	//// changed code ////

	__syncthreads() ; 

	if(blockSize >=512) 
	{
		if(tid < 256)
		{
			sdata[tid] += sdata[tid + 256] ; 
		}
		__syncthreads() ; 
	}

	if(blockSize >=256) 
	{
		if(tid < 128)
		{
			sdata[tid] += sdata[tid + 128] ; 
		}
		__syncthreads() ; 
	}

	if(blockSize >=128) 
	{
		if(tid < 64)
		{
			sdata[tid] += sdata[tid + 64] ; 
		}
		__syncthreads() ; 
	}

	if(blockSize >=32) 
	{
		warpReduce <blockSize > (sdata , tid) ; 	
	}

	if(tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0] ; 
	}

}


__global__ void reduce7(int *g_idata , int *g_odata , unsigned int n)
{
	int *sdata = SharedMemory <int> () ; 

	unsigned int tid = threadIdx.x ; 

	/// changed code ////
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x ; 

	unsigned int gridSize = blockSize*2*gridDim.x ;
	sdata[tid] = 0 ; 

	while(i<n)
	{
		sdata[tid] = g_idata[i] + g_idata[i+blockDim.x] ;
		i += gridSize ; 	
	}

	//// changed code ////

	__syncthreads() ; 

	for(unsigned int s=blockDim.x/2 ; s>32 ; s>>=1)
	{

		//// changed code /////

		if(tid < s)
		{
			sdata[index] += sdata[index + s] ; 
		}

		/////changed code ///////

		__syncthreads() ;
	}

	if(tid < 32) 
	{
		warpReduce(sdata , tid)  
	}

	if(tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0] ; 
	}
}



int main()
{
	// start with cudaMemcpyHostToDevice .....
	int threadsPerBlock = 64 ; 
	int old_blocks , blocks = (N / threadsPerBlock) / 2 ;

	blocks = (blocks == 0) ? 1 : blocks ; 
	old_blocks = blocks ; 

	while(blocks >0)
	{
		sum <<< blocks , threadsPerBlock >>> (devPtrA) ; 
		old_blocks = blocks ; 
		blocks = (blocks / threadsPerBlock) / 2 ; 
	} 

	if(blocks == 0 && old_blocks != 1)
	{
		sum <<< 1 , old_blocks/2 >>> (devPtrA) ; 
	} 

	///// end with cudaMemcyDeviceToHost......
}