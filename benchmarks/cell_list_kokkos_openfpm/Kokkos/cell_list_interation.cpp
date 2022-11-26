#include "util/cuda_launch.hpp"

#ifndef __NVCC__
#error "CAZZO"
#endif

#include <Kokkos_Core.hpp>
#include <cstdio>
#include "NN/CellList/cuda/CellList_gpu.hpp"
#include "NN/CellList/CellList.hpp"

using view_type = Kokkos::View<double * [3]>;


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


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  init_wrappers();

  printf("LayoutRight\n");

  {

    int N = 100;
    Kokkos::View<float*[3],Kokkos::LayoutLeft> pos ("position", N);
    Kokkos::View<float*[3],Kokkos::LayoutLeft> pos_ord ("position_reordered", N);

    Kokkos::View<float*,Kokkos::LayoutLeft> mass ("Mass", N);
    Kokkos::View<float*,Kokkos::LayoutLeft> mass_ord ("Mass reordered", N);

    for (int i = 0 ; i < N ; i++) {
        pos(i,0) = (float)rand() / RAND_MAX;
	pos(i,1) = (float)rand() / RAND_MAX;
	pos(i,2) = (float)rand() / RAND_MAX;

	mass(i) = (float)rand() / RAND_MAX;

	std::cout << pos(i,0) << "  "  << pos(i,1) << "  " << pos(i,2) << std::endl;
    }

    std::cout << "##########################################################################" << std::endl;

    openfpm::vector<aggregate<float[3]>,PtrMemory,memory_traits_inte> w_pos;
    {PtrMemory & ptr = *(new PtrMemory(&pos(0,0),N*sizeof(float)*3));
    w_pos.setMemory(ptr);
    w_pos.resize(N);}

    for (int i = 0 ; i < N ; i++) {
	    std::cout << w_pos.template get<0>(i)[0] << "  " << w_pos.template get<0>(i)[1] << "  " << w_pos.template get<0>(i)[2] << std::endl;
    }

    openfpm::vector<aggregate<float[3]>,PtrMemory,memory_traits_inte> w_pos_ord;
    {PtrMemory & ptr = *(new PtrMemory(&pos_ord(0,0),N*sizeof(float)*3));
    w_pos_ord.setMemory(ptr);
    w_pos_ord.resize(N);}

    openfpm::vector<aggregate<float>,PtrMemory,memory_traits_inte> w_mass;
    {PtrMemory & ptr = *(new PtrMemory(&mass(0),N*sizeof(float)));
    w_mass.setMemory(ptr);
    w_mass.resize(N);}

    openfpm::vector<aggregate<float>,PtrMemory,memory_traits_inte> w_mass_ord;
    {PtrMemory & ptr = *(new PtrMemory(&mass_ord(0),N*sizeof(float)));
    w_mass_ord.setMemory(ptr);
    w_mass_ord.resize(N);}


    mgpu::ofp_context_t context(mgpu::gpu_context_opt::no_print_props);

    SpaceBox<3,float> box({0.0f,0.0f,0.0f},{1.0f,1.0f,1.0f});
    CellList_gpu<3,float,CudaMemory> cl2(box,{10,10,10},2);
    cl2.construct(w_pos,w_pos_ord,w_mass,w_mass_ord,context,N);

    cl2.debug_deviceToHost();
    int ncount = 0;
    for (int i = 0 ; i < cl2.getNCells(); i++) 
    {
        ncount += cl2.getNelements(i);
    }

    std::cout << "N part: " << ncount << std::endl;

  }
  Kokkos::finalize();
}

