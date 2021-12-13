#include <iostream>
#include <math.h>
#include <stdlib.h> 

const size_t size_w = 1024;
const size_t size_h = 1024;

typedef unsigned mytype;
typedef mytype  arr_t[size_w];
const mytype A_val = 1;
const mytype B_val = 2;

struct ele
{
  double s;
  double v[2];
  double t[2][2];
};


//The array of structures approach
__global__ void read_write_lin(double * pos, ele * prp){

    unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;

    prp[p].s = pos[2*p] + pos[2*p+1];

    prp[p].v[0] = pos[2*p];
    prp[p].v[1] = pos[2*p+1];

    prp[p].t[0][0] = pos[2*p];
    prp[p].t[0][1] = pos[2*p+1];
    prp[p].t[1][0] = pos[2*p] + pos[2*p+1];
    prp[p].t[1][1] = pos[2*p+1] - pos[2*p];

    pos[2*p] += 0.01f;
    pos[2*p+1] += 0.01f;
}

int main(){

  double* in, in_host;
  ele* out;

  int N = 16777216;
  size_t sz_out = N * sizeof(ele);
  size_t sz_in = N * 2 * sizeof(double);

  cudaMalloc((void**) &in, sz_in);
  cudaMalloc((void**) &out, sz_out);

  in_host = (double *)malloc(sz_in);

  for(int i = 0; i < N; i++) {
    in_host[i] = i;
    in_host[i+1] = i + 100.0;
  }

  cudaMemcpy(in,in_host,sz_in,cudaMemcpyHostToDevice);

  int block_size = 1024;
  int grid_size = N / block_size;

  read_write_lin<<<grid_size,block_size>>>(in, out);
}
