// ############################## CPU PROGRAM #################################


// void matrix_mul_kernel(float *m , float *n , float *p , int n)
// {
// 	for(int i=0 ; i<n ; i++)
// 	{
// 		for(int j=0 ; j<n ; j++)
// 		{
// 			float pvalue = 0.0 ; 
// 			for(int k=0 ; k<n ; k++)
// 			{
// 				pvalue += M[i][K] * N[k][j] ; 
// 			}

// 			P[i][j] = Pvalue ;
// 		}
// 	}
// }



// ############################### GPU PROGRAM ###########################


#include<cuda.h>
#include<cuda_runtime.h>

__global__ void MatrixMulKernel(float* d_M , float* d_N , float* d_P , int N)
{
	int i = blockIdx.y *blockDim.y + threadIdx.y ; 
	int j = blockIdx.x *blockDim.x + threadIdx.x ;
	if((i<N) && (j<N))
	{
		float Pvalue = 0.0 ; 
		for(int k=0 ; k<N ; k++)
		{
			Pvalue += d_M[i*N + k] * d_N[k*N + K] ; 
		}

		d_P[i*N + j] = Pvalue ; 
	}
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