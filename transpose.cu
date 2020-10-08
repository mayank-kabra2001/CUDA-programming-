
#include<cuda.h>
#include<cuda_runtime.h>


// ######################### CPU ONLY ###############################

// void transposeHost(float *out , float *in , const int nx , const int ny)
// {
// 	for(int iy=0 ; iy<ny ; iy++)
// 	{
// 		for(int ix=0 ; ix<nx ; ix++)
// 		{
// 			out[ix*ny + iy] = in[iy*nx + ix] ; 
// 		}
// 	}
// }

// ############################ GPU ONLY ###############################


__global__ void transposeNaiveRow(float *out , float *in , const int nx , const int ny)
{
	unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x ; 
	unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y ;

	if(ix<nx && iy<ny) 
	{
		out[ix*ny + iy] = in[iy*nx + ix] ; 
	} 
}

int main(int argc , char **argv) 
{
	int dev = 0 ; 
	cudaDeviceProp deviceProp ; 
	CHECK(cudaGetDevceProperties(&deviceProp , dev)) ; 
	printf("%s starting transpose at " ,argv[0]) ; 
	printf("device %d : %s" , dev , deviceProp.name) ; 
	CHECK(cudaSetDevice(dev)) ; 

	int nx = 1<<13 ; 
	int ny = 1<<13 ;

	int iKernel = 0 ; 
	int blockx = 32 ; 
	int blocky = 32 ; 

	if(argc >1) iKernel = stoi(argv[1]) ; 

	size_t nBytes = nx*ny*sizeof(float) ; 

	dim3 block (blockx , blocky) ; 
	dim3 grid ((nx + block.x -1)/ block.x , (ny + block.y -1)/block.y) ; 

	float *h_A = (float*)Malloc(nBytes) ; 
	float *hostRef = (float*)Malloc(nBytes) ; 
	float *gpuRef = (float*)Malloc(nBytes) ;

	initialData(h_A , nx*ny) ; 

	float *d_A , *d_C ; 
	CHECK(cudaMalloc((float**)&d_A , nBytes)) ; 
	CHECK(cudaMalloc((float**)&d_C , nBytes)) ;

	CHECK(cudaMemcpy(d_A , h_A , nBytes , cudaMemcpyHostToDevice)) ; 

	void(*kernel)(float* , float* , int , int) ; 
	char *kernelName ;

	switch(iKernel)
	{
		case 0 : kernel = &transposeNaiveRow ; kernelName = "NAIVE ROW" ; break ; 
		case 1 : kernel = &transposeNaiveCol ; kernelName = "NAIVE COL" ; break ; 	
	} 

	kernel <<< grid , block >>> (d_C , d_A , nx , ny) ; 
	CHECK(cudaGetLastError()) ; 
	CHECK(cudaMemcpy(gpuRef , d_C , nBytes , cudaMemcpyDeviceToHost)) ; 

}
