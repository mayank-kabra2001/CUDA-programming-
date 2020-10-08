
#include<cuda.h>
#include<cuda_runtime.h>

#define TILE_DIM 32 
#define BLOCK_ROWS 32 

__ global___ void transposeCoalesced(float* odata , float* idata , const int nx , const int ny)
{
	__shared__ float tile[TILE_DIM][TILE_DIM] ; 

	int x = blockIdx.x * TILE_DIM + threadIdx.x ; 
	int y = blockIdx.y * TILE_DIM + threadIdx.y ;
	int width = gridDim.x * TILE_DIM ; 
	for(int j=0 ; j<TILE_DIM ; j++)
	{
		tile[threadIdx.y + j][threadIdx.x] = idata[(y+j)*width	+ x] ; 
	}

	__syncthreads() ; 

	x = blockIdx.y*TILE_DIM + threadIdx.x ; 
	y = blockIdx.x*TILE_DIM + threadIdx.y ; 

	for(int j=0 ; j<TILE_DIM ; j += BLOCK_ROWS)
	{
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j] ; 
	}
}

__ global___ void transposeFineGrained(float* odata , float* idata , const int width , const int height)
{
	__shared__ float tile[TILE_DIM][TILE_DIM] ; 

	int x = blockIdx.x * TILE_DIM + threadIdx.x ; 
	int y = blockIdx.y * TILE_DIM + threadIdx.y ;
	int index = xIndex + (y)*width ; 
	for(int j=0 ; j<TILE_DIM ; j++)
	{
		tile[threadIdx.y + j][threadIdx.x] = idata[index + j*width] ; 
	}

	__syncthreads() ; 

	for(int j=0 ; j<TILE_DIM ; j += BLOCK_ROWS)
	{
		odata[index + j*height] = tile[threadIdx.x][threadIdx.y + j] ; 
	}
}

__ global___ void transposeFineGrained(float* odata , float* idata , const int width , const int height)
{
	__shared__ float tile[TILE_DIM][TILE_DIM] ; 

	int x = blockIdx.x * TILE_DIM + threadIdx.x ; 
	int y = blockIdx.y * TILE_DIM + threadIdx.y ;
	int index_in = x + y*width ; 
	x = blockIdx.y*TILE_DIM + threadIdx.x ; 
	y = blockIdx.x*TILE_DIM + threadIdx.y ;
	int index_out = x +  y*height ; 

	for(int j=0 ; j<TILE_DIM ; j++)
	{
		tile[threadIdx.y + j][threadIdx.x] = idata[index_in + j*width] ; 
	}

	__syncthreads() ; 

	for(int j=0 ; j<TILE_DIM ; j += BLOCK_ROWS)
	{
		odata[index_out + j*height] = tile[threadIdx.y + j][threadIdx.x] ; 
	}
}

