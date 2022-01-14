#include <iostream>
#include <math.h>
#include <stdlib.h> 

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

//The structure of arrays approach
__global__ void read_write_inte(double * pos, double * prp0, double * prp1, double * prp2, unsigned int n_pos)
{
    unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;

    prp0[0*n_pos + p] = pos[0*n_pos + p] + pos[1*n_pos+p];

    prp1[0*n_pos + p] = pos[0*n_pos + p];
    prp1[1*n_pos + p] = pos[1*n_pos + p];

    prp2[0*n_pos*2+0*n_pos + p] = pos[0*n_pos + p];
    prp2[0*n_pos*2+1*n_pos + p] = pos[1*n_pos + p];
    prp2[1*n_pos*2+0*n_pos + p] = pos[0*n_pos + p] + 
                                  pos[1*n_pos + p];
    prp2[1*n_pos*2+1*n_pos + p] = pos[1*n_pos + p] - 
                                  pos[0*n_pos + p];

    pos[0*n_pos + p] += 0.01f;
    pos[1*n_pos + p] += 0.01f;
}


int main(){

  double* in, *in_host;
  ele* out, *out_host;

  int N = 16777216;
  size_t sz_out = N * sizeof(ele);
  size_t sz_in = N * 2 * sizeof(double);

  cudaMalloc((void**) &in, sz_in);
  cudaMalloc((void**) &out, sz_out);

  in_host = (double *)malloc(sz_in);

  for(int i = 0; i < N; i++) {
    in_host[2*i] = i;
    in_host[2*i + 1] = i + 100.0;
  }

  cudaMemcpy(in,in_host,sz_in,cudaMemcpyHostToDevice);

  int block_size = 1024;
  int grid_size = N / block_size;

  read_write_lin<<<grid_size,block_size>>>(in, out);

  out_host = (ele *)malloc(sz_out);

  cudaMemcpy(out_host,out,sz_out,cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i ++) {
    if(out_host[i].s != (in_host[2 * i] + in_host[2 * i + 1])) {
      printf("The test has failed\n");
    }
  }

  double *prp0, *prp0_host, *prp1, *prp2, *pos, *pos_host;

  cudaMalloc((void**) &prp0, N * sizeof(double));
  cudaMalloc((void**) &prp1, 2 * N * sizeof(double));
  cudaMalloc((void**) &prp2, 4 * N * sizeof(double));
  cudaMalloc((void**) &pos, 2 * N * sizeof(double));

  pos_host = (double *) malloc(2 * N * sizeof(double));

  for(int i = 0; i < N; i++) {
    pos_host[i] = i;
    pos_host[N+i] = i + 100.0;
  }

  cudaMemcpy(pos,pos_host,2 * N * sizeof(double),cudaMemcpyHostToDevice);

  read_write_inte<<<grid_size,block_size>>>(pos, prp0, prp1, prp2, N);

  prp0_host = (double *)malloc(N * sizeof(double));

  cudaMemcpy(prp0_host,prp0,N * sizeof(double),cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i ++) {

    bool c1 = (prp0_host[i] == pos_host[i] + pos_host[N + i]);

    if(!c1) {
      printf("The test has failed\n");
    }
  }

}
