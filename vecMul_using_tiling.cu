#include<cuda.h>
#include<cuda_runtime.h>


__global__ void MatrixMulKernel(float *d_m , float *d_N , float *d_P , int width)
{
	__shared__ float Mds(TILE_WIDTH)[TILE_WIDTH] ; 
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH] ; 

	int bx = blockIdx.x ; 
	int by = blockIdx.y ; 
	int tx = threadIdx.x ; 
	int ty = threadIdx.y ; 

	int Row = by*TILE_WIDTH + ty ; 
	int Col = bx*TILE_WIDTH + tx ; 
	float Pvalue = 0  ; 
	for(int m=0 ; m<width/TILE_WIDTH ; m++)
	{
		Mds[ty][tx] = d_M[Row*width + m*TILE_WIDTH] ; 
		Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*width + Col] ;
		__syncthreads() ; 
		
		for(int k=0 ; k<TILE_WIDTH ; k++)
		{
			Pvalue += Mds[ty][k] * Nds[k][tx] ; 
		} 
		__syncthreads() ; 
	}
	d_P[Row*width + Col] = Pvalue ; 
}

int main()
{
	int size = 16*16 ; 
	cudaMemcpy (d_M , M , size*sizeof(float) , cudaMemHostToDevice) ;
	cudaMemcpy (d_N , N , size*sizeof(float) , cudaMemHostToDevice) ;

	dim3 grid(2,2,1) ; 
	dim3 block(8,8,1) ;

	int N = 16 ; 
	MatrixMulKernel <<< grd , block >>> (d_M , d_N , d_P , P) ; 
	cudaMemcpy(p , d_P , size*sizeof(float) , cudaMemDeviceToHost) ;
	  
}

