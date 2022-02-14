#include <Kokkos_Core.hpp>
#include <cstdio>

using view_type = Kokkos::View<double * [3]>;

#define NELEMENTS 67108864

static inline void standard_deviation(std::vector<double> measures, double & mean, double & dev)
{
	mean = 0;
	for (size_t i = 0 ; i < measures.size() ; i++)
		mean += measures[i];
	mean /= measures.size();

	dev = 0;
	for (size_t i = 0 ; i < measures.size() ; i++)
		dev += (measures[i] - mean)*(measures[i] - mean);

	dev = sqrt(dev / (measures.size() - 1));
}

struct ele
{
  double s;
  double v[2];
  double t[2][2];
};

template<typename vd_out_prp0_type, typename vd_out_prp1_type, typename vd_out_prp2_type, typename vd_in_type>
void struct_of_arrays_read (vd_out_prp0_type vd_out_prp0, vd_out_prp1_type vd_out_prp1, vd_out_prp2_type vd_out_prp2, vd_in_type vd_in, int N, std::vector<double> & times) {

   for (int i = 0 ; i < 110 ; i++)
   {
      Kokkos::fence();
      Kokkos::Timer timer;

      Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {

          float a = vd_out_prp0(i);

          float b = vd_out_prp1(i, 0);
          float c = vd_out_prp1(i, 1);

          float d = vd_out_prp2(i,0,0);
          float e = vd_out_prp2(i,0,1);
          float f = vd_out_prp2(i,1,0);
          float g = vd_out_prp2(i,1,1);

	  float h = vd_in(i,0);
	  vd_in(i,1) = a+b+c+d+e+f+g+h;

        });

        Kokkos::fence();
        double time = timer.seconds();
        std::cout << "StructOfArraysKokkos READ: " << time  << "  " << 1.*N*4*9 / time * 1e-9 << " GB/s"  << std::endl;

        times.push_back(1.*N*4*9 / time * 1e-9);
  }
}

template<typename vd_out_prp0_type, typename vd_out_prp1_type, typename vd_out_prp2_type, typename vd_in_type>
void struct_of_arrays_write (vd_out_prp0_type vd_out_prp0, vd_out_prp1_type vd_out_prp1, vd_out_prp2_type vd_out_prp2, vd_in_type vd_in, int N, std::vector<double> & times) {

   for (int i = 0 ; i < 110 ; i++)
   {

      Kokkos::fence();
      Kokkos::Timer timer;

      Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {

          float a = vd_in(i, 0);

          vd_out_prp0(i) = a;

          vd_out_prp1(i, 0) = a;
          vd_out_prp1(i, 1) = a;

          vd_out_prp2(i, 0, 0) = a;
          vd_out_prp2(i, 0, 1) = a;
          vd_out_prp2(i, 1, 0) = a;
          vd_out_prp2(i, 1, 1) = a;

          vd_in(i, 1) = a;

        });

        Kokkos::fence();
        double time = timer.seconds();
        std::cout << "StructOfArraysKokkos WRITE: " << time  << "  " << 1.*N*4*9 / time * 1e-9 << " GB/s"  << std::endl;

        times.push_back(1.*N*4*9 / time * 1e-9);
   }

}

#ifdef KOKKOS_ENABLE_CUDA
__global__ void kernel(int N, float* vd_in, float* vd_out_prp0, float* vd_out_prp1, float* vd_out_prp2) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
  float a = vd_in[i];

  vd_out_prp0[i] = a;

  vd_out_prp1[i] = a;
  vd_out_prp1[i + N] = a;

  vd_out_prp2[i] = a;
  vd_out_prp2[i + 2*N] = a;
  vd_out_prp2[i + N] = a;
  vd_out_prp2[i + 3*N] = a;

  vd_in[i + N] = a;
  }
}

template<typename vd_out_prp0_type, typename vd_out_prp1_type, typename vd_out_prp2_type, typename vd_in_type>
void struct_of_arrays_cuda (vd_out_prp0_type vd_out_prp0, vd_out_prp1_type vd_out_prp1, vd_out_prp2_type vd_out_prp2, vd_in_type vd_in, int N) {

      Kokkos::fence();
      Kokkos::Timer timer;
      int blocks = (N+255)/256;
      kernel<<<blocks,256>>>(N,vd_in.data(),vd_out_prp0.data(),vd_out_prp1.data(),vd_out_prp2.data());
      Kokkos::fence();
      double time = timer.seconds();
      std::cout << "StructOfArraysCuda: " << time  << "  " << 1.*N*8*9 / time * 1e-9 << " GB/s"  << std::endl;
}
#endif

template<typename prp_type, typename pos_type>
void array_of_structs (prp_type prp, pos_type pos, int N) {

        Kokkos::fence();
        Kokkos::Timer timer;

        Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
          auto pos_i_0 = pos(i, 0);
          auto pos_i_1 = pos(i, 1);
          prp(i).s = pos_i_0 + pos_i_1;

          prp(i).v[0] = pos_i_0;
          prp(i).v[1] = pos_i_1;


          prp(i).t[0][0] = pos_i_0;
          prp(i).t[0][1] = pos_i_1;
          prp(i).t[1][0] = pos_i_0 + pos_i_1;
          prp(i).t[1][1] = pos_i_1 - pos_i_0;

          pos(i, 0) = pos_i_0 + 0.01;
          pos(i, 1) = pos_i_1 + 0.01;

        });

        Kokkos::fence();
        double time = timer.seconds();
        std::cout << "TIME: " << time << "  " << (double)N*8*11 / time * 1e-9 << " GB/s"  << std::endl;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  printf("LayoutRight\n");

  {

    int N = NELEMENTS;
    Kokkos::View<float*[2]> pos ("position", N);
    Kokkos::View<float*> prp_s ("scalar_property", N);
    Kokkos::View<float*[2]> prp_v ("vector_property", N);
    Kokkos::View<float*[2][2]> prp_t ("tensor_property", N);

    Kokkos::View<ele*> prp ("aos_prp", N);

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
          pos(i, 0) = i;
          pos(i, 1) = i + 100.0;
        });

    std::vector<double> times_read;
    std::vector<double> times_write;

    struct_of_arrays_write(prp_s, prp_v, prp_t, pos, N, times_read);
    struct_of_arrays_read(prp_s, prp_v, prp_t, pos, N, times_write);
//    array_of_structs(prp, pos, N);

    double mean_read;
    double mean_write;

    double dev_read;
    double dev_write;

    standard_deviation(times_read,mean_read,dev_read);
    standard_deviation(times_write,mean_write,dev_write);

    std::cout << "READ MEAN: " << mean_read  << " DEV: " << dev_read  << std::endl;
    std::cout << "WRITE MEAN: " << mean_write  << " DEV: " << dev_write  << std::endl;

#else

    printf("Error: While building Kokkos, the option -DKokkos_ENABLE_CUDA_LAMBDA=true needs to be specified.");

#endif
  }
  printf("LayoutLeft\n");

  Kokkos::finalize();
}

