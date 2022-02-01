
#ifdef __NVCC__

#define PRINT_STACKTRACE
//#define STOP_ON_ERROR
#define OPENMPI
//#define SE_CLASS1

//#define USE_LOW_REGISTER_ITERATOR
#define SCAN_WITH_CUB //<------ In case you want to use CUB for scan operations
#define SORT_WITH_CUB
//#define EXTERNAL_SET_GPU <----- In case you want to distribute the GPUs differently from the default

#include "Vector/vector_dist.hpp"
#include <math.h>
#include "Draw/DrawParticles.hpp"
#include "util/stat/common_statistics.hpp"

template<typename T>
__global__ void kernel_test(T p, float * array)
{
	__shared__ int cnt;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	cnt = 0;

	__syncthreads();
	atomicAdd(&cnt,1);
	__syncthreads();

	array[tid] = cnt;
}


template<typename T>
__global__ void test1_syncthreads(T p, float * array)
{
    __shared__ int cnt;

    cnt = 0;

    size_t idx_x = blockIdx.x + threadIdx.x;
    size_t idx_y = blockIdx.y + threadIdx.y;
    size_t idx_z = blockIdx.z + threadIdx.z;

    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();

    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);


    array[/*idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x*/0] = cnt + idx_x + idx_y + idx_z;
}

template<typename T>
__global__ void test1_no_syncthreads(T p, float * array)
{
    __shared__ int cnt;

    cnt = 0;

    size_t idx_x = blockIdx.x + threadIdx.x;
    size_t idx_y = blockIdx.y + threadIdx.y;
    size_t idx_z = blockIdx.z + threadIdx.z;

//    array[/*idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x*/0] = idx_x + idx_y + idx_z/*64*/;

    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);

//    p.ptr[idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x] = cnt + idx_x + idx_y + idx_z;

    array[0] = cnt + idx_x + idx_y + idx_z;
}

template<typename T>
__global__ void test1_only_tls(T p, float * array)
{
    __shared__ int cnt;

    cnt = 0;

    size_t idx_x = blockIdx.x + threadIdx.x;
    size_t idx_y = blockIdx.y + threadIdx.y;
    size_t idx_z = blockIdx.z + threadIdx.z;

    array[0] += idx_x + idx_y + idx_z;

}


struct par_struct
{
    float * ptr;
};

struct ite_g
{
    dim3 wthr;
    dim3 thr;

    size_t nblocks()
	{
		return wthr.x * wthr.y * wthr.z;
	}

	size_t nthrs()
	{
		return thr.x * thr.y * thr.z;
	}
};

int main(int argc, char* argv[])
{

    // initialize the library
	openfpm_init(&argc,&argv);

	openfpm::vector<double> tele_lamb;
	openfpm::vector<double> tele_ker;
	openfpm::vector<double> tele_no_sync;
	openfpm::vector<double> tele_tls;

	for (int i = 0; i < 10; i++)
	{

	    CudaMemory mem;
	    mem.allocate(256*256*256*sizeof(float));

	    CudaMemory mem2;
	    mem2.allocate(256*256*256*sizeof(float));

	    float * array_ptr = (float *)mem.getDevicePointer();

	    par_struct p;
	    p.ptr = (float *)mem2.getDevicePointer();

	    ite_g g;

	    float * array = array_ptr;

	    g.wthr = dim3(64,64,64);
	    g.thr = dim3(4,4,4);

	    timer t_lamb;
	    t_lamb.start();

	    auto lambda_f = [array,p] __device__ (dim3 & blockIdx, dim3 & threadIdx){
       		__shared__ int cnt;

        	cnt = 0;
    
        	size_t idx_x = blockIdx.x + threadIdx.x;
        	size_t idx_y = blockIdx.y + threadIdx.y;
        	size_t idx_z = blockIdx.z + threadIdx.z;
    
    
        	__syncthreads();
        	__syncthreads();
        	__syncthreads();
        	__syncthreads();
        	__syncthreads();
        	__syncthreads();
        	__syncthreads();
        	__syncthreads();
    
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    atomicAdd(&cnt,1);
    
        	array[/*idx_z*gridDim.x*gridDim.y*blockDim.x*blockDim.y + idx_y*gridDim.x*blockDim.x + idx_x*/0] = cnt + idx_x + idx_y + idx_z;
    	    };

    	    CUDA_LAUNCH_LAMBDA(g, lambda_f);

	    t_lamb.stop();

	    std::cout << "TLAMBDA : "  <<  t_lamb.getwct() << std::endl;

	    tele_lamb.add(t_lamb.getwct());


    	    mem.deviceToHost();
    	    mem2.deviceToHost();

    	    float * ptr1 = (float *)mem.getPointer();
    	    float * ptr2 = (float *)mem2.getPointer();

    	    bool check = true;
    	    for (int i = 0 ; i < 16*16*16; i++)
    	    {
        	//std::cout << i << "   " << ptr1[i] << "  " << ptr2[i] << std::endl;

        	check &= ptr1[i] == 64.0;
        	check &= ptr2[i] == 128.0;
    	    }

//    	    if (check != true)
//	    {std::cout << "FAILED" << std::endl; exit(0);}

        //////////////////////////////////////////////////////////////////////////////////////////////////


            g.wthr = dim3(64,64,64);
            g.thr = dim3(4,4,4);

            timer t_ker;
            t_ker.start();

            CUDA_LAUNCH(test1_syncthreads,g,p,array_ptr);

            t_ker.stop();

	    std::cout << "TKERNEL: " << t_ker.getwct() << std::endl;

            tele_ker.add(t_ker.getwct());

            mem.deviceToHost();
            mem2.deviceToHost();

	    {
            float * ptr1 = (float *)mem.getPointer();
            float * ptr2 = (float *)mem2.getPointer();

            bool check = true;
            for (int i = 0 ; i < 128*128*128; i++)
            {
                //std::cout << i << "   " << ptr1[i] << "  " << ptr2[i] << std::endl;

                check &= ptr1[i] == 64.0;
                check &= ptr2[i] == 128.0;
            }

//            if (check != true)
//            {std::cout << "FAILED" << std::endl; exit(0);}
	    }


	//////////////////////////////////////////////////////////////////////////////////////////////////

    	    g.wthr = dim3(64,64,64);
    	    g.thr = dim3(4,4,4);

	    {
            timer t_ker;
            t_ker.start();

            CUDA_LAUNCH(test1_no_syncthreads,g,p,array_ptr);

            t_ker.stop();

	    tele_no_sync.add(t_ker.getwct());
	    }

            mem.deviceToHost();
            mem2.deviceToHost();

	    { 
            float * ptr1 = (float *)mem.getPointer();
            float * ptr2 = (float *)mem2.getPointer();

            bool check = true;
            for (int i = 0 ; i < 256*256*256; i++)
            {
                //std::cout << i << "   " << ptr1[i] << "  " << ptr2[i] << std::endl;

                check &= ptr1[i] == 64.0;
                check &= ptr2[i] == 128.0;
            }
	    }

//            if (check != true)
//            {std::cout << "FAILED" << std::endl; exit(0);}

	    timer t_tls;
            t_tls.start();

            CUDA_LAUNCH(test1_only_tls,g,p,array_ptr);

            t_tls.stop();
            tele_tls.add(t_tls.getwct());


      ///////////////////////////////////////////////////////////////////////////////////////////////////////

	}

	double tele_lamb_mean;
	double tele_lamb_dev;
	standard_deviation(tele_lamb,tele_lamb_mean,tele_lamb_dev);

	double tele_ker_mean;
        double tele_ker_dev;
	standard_deviation(tele_ker,tele_ker_mean,tele_ker_dev);

        double tele_ns_ker_mean;
        double tele_ns_ker_dev;
        standard_deviation(tele_no_sync,tele_ns_ker_mean,tele_ns_ker_dev);

	double tele_tls_mean;
	double tele_tls_dev;
	standard_deviation(tele_tls,tele_tls_mean,tele_tls_dev);

	std::cout << tele_lamb_mean << " " << tele_ker_mean << " " << tele_ns_ker_mean << "  "  << tele_tls_mean << std::endl;

	std::cout << "SYNCTHREAD LATENCY: " << (tele_ker_mean - tele_ns_ker_mean) / (256*256*256*8) * 1e9 << " ns " << " error: " << tele_ker_dev + tele_ns_ker_dev << std::endl;
	std::cout << "ATOMIC ADD: " << (tele_ns_ker_mean - tele_tls_mean) * 1e9 / (256.0*256.0*256.0*240.0) << " ns " << " error: " << tele_ns_ker_dev + tele_tls_dev << std::endl;
	std::cout << "TLS: " << (tele_ker_mean - tele_lamb_mean) * 1e9 / (256*256*256*6) << "ns " << " error: " << tele_lamb_dev + tele_ker_dev << std::endl;

	openfpm_finalize();
}
 
#else

int main(int argc, char* argv[])
{
        return 0;
}

#endif
