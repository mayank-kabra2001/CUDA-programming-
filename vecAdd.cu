//#####################		VECTOR ADDITION CPU ONLY  	########################

// #include<stdio.h>

// void vecAdd(float* h_A , float* h_B , float* h_C , int n )
// {
// 	for(int i=0 ; i<n ; i++)
// 	{
// 		h_C[i] = h_A[i] + h_B[i] ; 
// 	}
// }

// int main()
// {
// 	float *h_A , *h_B , *h_C ; 
// 	int n ; 
// 	h_A = (float*)malloc(n*sizeof(float)) ; 
// 	h_B = (float*)malloc(n*sizeof(float)) ; 
// 	h_C = (float*)malloc(n*sizeof(float)) ; 
// 	vecAdd(h_A , h_B , h_C , n) ; 
// }


// #######################		VECTOR ADDITION CPU - GPU 	######################

#include<cuda.h>
#include<cuda_runtime.h>

__global__ void vectorAdd(float* , float* , float* , int)

__global__ void vectorAdd(float*A , float*B , float*C , int n)
{
	// cuda kernel definition 
	int i = threadIdx.x + blockDim.x*blockIdx.x ; 
	if(i<n) 
	{
		C[i] = A[i] + B[i] ; 
	}
}

void vecAdd(float*A , float*B , float*C , int n)
{
	//host progress 
	int size = n*sizeof(float) ; 
	float *d_A = NULL , *d_B = NULL , *d_C = NULL ; 

	cudaError_t err = cudaSuccess ; 

	err = cudaMalloc((void**)&d_A , size) ; 
	if(err != cudaSuccess)
	{
		fprintf(stderr , " Failed to allocate device vector A (error code %s)!\n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}

	err = cudaMalloc((void**)&d_B , size) ; 
	if(err != cudaSuccess)
	{
		fprintf(stderr , " Failed to allocate device vector B (error code %s)!\n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}

	err = cudaMalloc((void**)&d_C , size) ; 
	if(err != cudaSuccess)
	{
		fprintf(stderr , " Failed to allocate device vector C (error code %s)!\n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}

	printf("copy input data from the host memory to the CUDA device\n") ; 
	
	err = cudaMemcpy(d_A , h_A , size , cudaMemcpyHostToDevice) ; 
	if(err != cudaSuccess)
	{
		fprintf(stderr , " Failed  to copy vector A from host to device (error code %s)!\n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}
	
	err = cudaMemcpy(d_B , h_B , size , cudaMemcpyHostToDevice) ; 
	if(err != cudaSuccess)
	{
		fprintf(stderr , " Failed  to copy vector B from host to device (error code %s)!\n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}
	
	err = cudaMemcpy(d_C , h_C , size , cudaMemcpyHostToDevice) ; 
	if(err != cudaSuccess)
	{
		fprintf(stderr , " Failed  to copy vector C from host to device (error code %s)!\n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}

	int threadsPerBlock = 256 ; 
	int blockPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock ; 
	printf("CUDA kernel launch with %d blocks of %d threads \n" , threadsPerBlock , blockPerGrid) ; 

	vectorAdd <<< blockPerGrid , threadsPerBlock >>> (d_A , d_B , d_C , n) ; 
	err = cudaGetLastError() ; 

	if(err != cudaSuccess)
	{
		fprintf(stderr , "Failed to launch vectorAdd kernel (error code %s)!\n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}

	printf("copy output data from the output device to the host memcpy \n") ; 
	err = cudaMemcpy(h_C , d_C , size , cudaMemcpyDeviceToHost) ; 
	if(err != cudaSuccess) 
	{
		fprintf(stderr , "Failed to copy vector C from device to host (error code %s)! \n" , cudaGetErrorString(err)) ; 
		exit(EXIT_FAILURE) ; 
	}

	cudaFree(d_A) ; cudaFree(d_B) ; cudaFree(d_C) ;

	for(int i=0 ; i<n ; i++)
	{
		if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr , " Result verification failed at element %d !\n" , err) ; 
			exit(EXIT_FAILURE) ;  
		}
	}

	printf("TEST PASSED") ; 
}

int main()
{
	float *h_A , *h_B , *h_C ; 
	int n ; 
	h_A = (float*)malloc(n*sizeof(float)) ; 
	h_B = (float*)malloc(n*sizeof(float)) ; 
	h_C = (float*)malloc(n*sizeof(float)) ; 
	vecAdd(h_A , h_B , h_C , n) ; 
}