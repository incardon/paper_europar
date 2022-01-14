#include <stdio.h>
#include <omp.h>
#include <cstdlib>

struct ele
{
  double s;
  double v[2];
  double t[2][2];
};


void array_of_structs (double * pos, ele * prp, int N)  {

  #pragma omp parallel for
  for(int p = 0; p < N; p++) {

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

}


void struct_of_arrays (double *prp0, double *prp1, double *prp2, double *pos, int N)  {

    int n_pos = N;

    #pragma omp parallel for
    for(int p = 0; p < N; p++) {

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

}


int main(int argc, char** argv){

  bool aos = false;
  bool soa = false;

  if(argc == 0) {

    printf("To run AOS only, use runtime argument 1. To run SOA only, use 2. Running both...\n");
    aos = true;
    soa = true;
  } else {
    long ch = strtol(argv[1], NULL, 10);

    if(ch == 1) {
        aos = true;
    } else if(ch == 2) {
        soa = true;
    } else {
        printf("To run AOS only, use runtime argument 1. To run SOA only, use 2. For both, leave the argument empty. Exiting...\n");
    }
  }

  int N = 16777216;

  if(aos) {

    double* pos;
    ele* prp;

    size_t sz_out = N * sizeof(ele);
    size_t sz_in = N * 2 * sizeof(double);

    pos = (double *) malloc(sz_in);
    prp = (ele *) malloc(sz_out);

    for(int i = 0; i < N; i++) {
      pos[2*i] = i;
      pos[2*i + 1] = i + 100.0;
    }     
    
    array_of_structs(pos, prp, N);

    //Error checking //TODO: complete for other properties
    for(int i = 0; i < N; i ++) {
      if(prp[i].s != (pos[2 * i] + pos[2 * i + 1] - 0.02)) {
        printf("In AOS, the test has failed\n");
        printf("Index: %d. Value should have been: %lf but was %lf\n", i, (pos[2 * i] + pos[2 * i + 1] - 0.02), prp[i].s);
      
        break;
      }
    }
  }

  if(soa) {
    double *prp0, *prp0_host, *prp1, *prp2, *pos;

    prp0 = (double *)malloc(N * sizeof(double));
    prp1 = (double *)malloc(2 * N * sizeof(double));
    prp2 = (double *)malloc(4 * N * sizeof(double));
    pos = (double *)malloc(2 * N * sizeof(double));

    for(int i = 0; i < N; i++) {
        pos[i] = i;
        pos[N+i] = i + 100.0;
    }

    struct_of_arrays(prp0, prp1, prp2, pos, N);
    
    //Error checking //TODO: complete for other properties
    for(int i = 0; i < N; i ++) {
      bool c1 = (prp0[i] == pos[i] + pos[N + i]);
      if(!c1) {
        printf("In SOA, the test has failed\n");
        break;
      }
    }

  }

}