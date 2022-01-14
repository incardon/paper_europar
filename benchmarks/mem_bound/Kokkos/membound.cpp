#include <Kokkos_Core.hpp>
#include <cstdio>

using view_type = Kokkos::View<double * [3]>;

struct ele
{
  double s;
  double v[2];
  double t[2][2];
};

void struct_of_arrays (Kokkos::View<double*> prp0, Kokkos::View<double*[2]> prp1, Kokkos::View<double*[2][2]> prp2, Kokkos::View<double*[2]> pos, int N) {

      Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
          prp0(i) = pos(i, 0) + pos(i, 1);
         
          prp1(i, 0) = pos(i, 0);
          prp1(i, 1) = pos(i, 1);

          prp2(i, 0, 0) = pos(i, 0);
          prp2(i, 0, 1) = pos(i, 1);
          prp2(i, 1, 0) = pos(i, 0) + pos(i, 1);
          prp2(i, 1, 1) = pos(i, 1) - pos(i, 0);

          pos(i, 0) += 0.01;
          pos(i, 1) += 0.01;

        });
}

void array_of_structs (Kokkos::View<ele*> prp, Kokkos::View<double*[2]> pos, int N) {

        Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
          prp(i).s = pos(i, 0) + pos(i, 1);
         
          prp(i).v[0] = pos(i, 0);
          prp(i).v[1] = pos(i, 1);

          prp(i).t[0][0] = pos(i, 0);
          prp(i).t[0][1] = pos(i, 1);
          prp(i).t[1][0] = pos(i, 0) + pos(i, 1);
          prp(i).t[1][1] = pos(i, 1) - pos(i, 0);

          pos(i, 0) += 0.01;
          pos(i, 1) += 0.01;

        });
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  printf("Hello world\n");

  {

    int N = 16777216;
    Kokkos::View<double*[2]> pos ("position", N);
    Kokkos::View<double*> prp_s ("scalar_property", N);
    Kokkos::View<double*[2]> prp_v ("vector_property", N);
    Kokkos::View<double*[2][2]> prp_t ("tensor_property", N);

    Kokkos::View<ele*> prp ("aos_prp", N);
  
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    
    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(const int i) {
          pos(i, 0) = i;
          pos(i, 1) = i + 100.0;
        });

    struct_of_arrays(prp_s, prp_v, prp_t, pos, N);
    array_of_structs(prp, pos, N);


#else
    
    printf("Error: While building Kokkos, the option -DKokkos_ENABLE_CUDA_LAMBDA=true needs to be specified.");

#endif
  }
  Kokkos::finalize();
}