
// ########################### CPU JULIA FXN ###############################

// struct complex
// {
// 	float r ; 
// 	float i ; 
// } ;

// float magnitude (struct complex a) 
// {
// 	return ((a.r*a.r) + (a.i*a.i)) ;
// }

// void add(struct complex a , struct complex b)
// {
// 	res->r = a.r + b.r ; 
// 	res->i = a.i + b.i ; 
// }

// void mul(struct complex a , struct complex b) 
// {
// 	res->r = (a.r*b.r) + (a.i*b.i) ; 
// 	res->i = (a.r*b.i) + (a.i*b*r) ; 
// }

// int julia (int x , int y)
// {
// 	const float scale 1.5 ; 
// 	float jx = scale*(float)(DIM/2 - x)(DIM/2) ;
// 	float jy = scale*(float)(DIM/2 - x)(DIM/2) ;

// 	struct complex c , a , r1 , r2 ; 

// 	c.r = -0.8 ; c.i = 0.154 ; 
// 	a.r = jx ; a.i = jy ; 

// 	int i=0 ; 
// 	for(int i=0 ; i<200 ; i++)
// 	{
// 		mul(a , a , &r1) ; 
// 		add(r1 , c , &r2) ; 
// 		if(magnitude(r2) > 1000)
// 			return 0 ; 
// 		a.r = r2.r ; 
// 		a.i = r2.i ; 
// 	}

// 	return 0 ; 
// }


// void kernel (unsigned char *ptr) 
// {
// 	for(int y=0 ; y<DIM ; y++)
// 	{
// 		for (int x = 0; x <DIM ; x++)
// 		{
// 			int offset = x+ y*DIM ; 
// 			int juliaValue = julia(x , y) ; 
// 			ptr [offset*4 + 0] = 255*juliaValue ;
// 			ptr [offset*4 + 1] = 0 ; 
// 			ptr [offset*4 + 2] = 0 ; 
// 			ptr [offset*4 + 3] = 255 ; 
//  		}
// 	}
// }


// int main(void) 
// {
// 	CPUBITmap bitmap(DIM , DIM) ; 
// 	unsigned char *ptr = bitmap.get_ptr() ; 
// 	kernel(ptr) ; 
// 	bitmap.dispay_and_exit() ; 
// }



// ###########################  GPU JULIA FXN   #####################################

#include<cuda.h>
#include<cuda_runtime.h>


struct cucomplex
{
	float r ; 
	float i ; 
} ;

__device__ float magnitude (struct complex a) 
{
	return ((a.r*a.r) + (a.i*a.i)) ;
}

__device__ void add(struct complex a , struct complex b)
{
	res->r = a.r + b.r ; 
	res->i = a.i + b.i ; 
}

__device__ void mul(struct complex a , struct complex b) 
{
	res->r = (a.r*b.r) + (a.i*b.i) ; 
	res->i = (a.r*b.i) + (a.i*b*r) ; 
}

__device__ int julia (int x , int y)
{
	const float scale 1.5 ; 
	float jx = scale*(float)(DIM/2 - x)(DIM/2) ;
	float jy = scale*(float)(DIM/2 - x)(DIM/2) ;

	struct complex c , a , r1 , r2 ; 

	c.r = -0.8 ; c.i = 0.154 ; 
	a.r = jx ; a.i = jy ; 

	int i=0 ; 
	for(int i=0 ; i<200 ; i++)
	{
		mul(a , a , &r1) ; 
		add(r1 , c , &r2) ; 
		if(magnitude(r2) > 1000)
			return 0 ; 
		a.r = r2.r ; 
		a.i = r2.i ; 
	}

	return 0 ; 
}


__global__ void kernel (unsigned char *ptr) 
{
	int x = blockIdx.x ; 
	int y = blockIdx.y ; 

	int offset = x + y*gridDim.x ;  
	int juliaValue = julia(x , y) ; 
	ptr [offset*4 + 0] = 255*juliaValue ;
	ptr [offset*4 + 1] = 0 ; 
	ptr [offset*4 + 2] = 0 ; 
	ptr [offset*4 + 3] = 255 ; 
}


int main(void) 
{
	CPUBITmap bitmap(DIM , DIM) ; 
	unsigned char *dev_bitmap ; 

	cudaMalloc((void**) &dev_bitmap , bitmap.image_size())
	dim3 grid(DIM , DIM) ; 
	kernel <<< grid , 1 >>> (dev_bitmap) ; 

	cudaMemcpy(bitmap.get_ptr() , dev_bitmap , bitmap.image_size() , cudaMemcpyDeviceToHost)  ; 
	bitmap.dispay_and_exit() ; 
	cudaFree(dev_bitmap) ;
}

