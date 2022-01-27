/*! \page Vector_7_sph_dlb_gpu_opt Vector 7 SPH Dam break simulation with Dynamic load balacing on Multi-GPU (optimized version)
 *
 *
 * [TOC]
 *
 *
 * # SPH with Dynamic load Balancing on GPU (Optimized) # {#SPH_dlb_gpu_opt}
 *
 *
 * This example show the classical SPH Dam break simulation with load balancing and dynamic load balancing. The main difference with
 * \ref{SPH_dlb} is that here we use GPU and 1.2 Millions particles. Simulate 1.5 second should be duable on a 1050Ti within a couple
 * of hours.
 *
 * \htmlonly
 * <a href="#" onclick="hide_show('vector-video-3')" >Simulation video 1</a><br>
 * <div style="display:none" id="vector-video-3">
 * <video id="vid3" width="1200" height="576" controls> <source src="http://openfpm.mpi-cbg.de/web/images/examples/7_SPH_dlb/sph_gpu1.mp4" type="video/mp4"></video>
 * </div>
 * <a href="#" onclick="hide_show('vector-video-4')" >Simulation video 2</a><br>
 * <div style="display:none" id="vector-video-4">
 * <video id="vid4" width="1200" height="576" controls> <source src="http://openfpm.mpi-cbg.de/web/images/examples/7_SPH_dlb/sph_gpu2.mp4" type="video/mp4"></video>
 * </div>
 * <a href="#" onclick="hide_show('vector-video-15')" >Simulation video 3</a><br>
 * <div style="display:none" id="vector-video-15">
 * <video id="vid15" width="1200" height="576" controls> <source src="http://openfpm.mpi-cbg.de/web/images/examples/7_SPH_dlb/sph_gpu3.mp4" type="video/mp4"></video>
 * </div>
 * \endhtmlonly
 *
 *
 * ## GPU ## {#e7_sph_inclusion}
 *
 * This example is an optimization of the example \ref SPH_dlb_gpu all the optimization operated on this example has been explained
 * here \ref e3_md_gpu_opt so we will not go into the details
 *
 * we report the full code here
 *
 *
 */

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


typedef float real_number;

// A constant to indicate boundary particles
#define BOUNDARY 0

// A constant to indicate fluid particles
#define FLUID 1

// initial spacing between particles dp in the formulas
const real_number dp = 0.00425;
// Maximum height of the fluid water
// is going to be calculated and filled later on
real_number h_swl = 0.0;

// c_s in the formulas (constant used to calculate the sound speed)
const real_number coeff_sound = 20.0;

// gamma in the formulas
const real_number gamma_ = 7.0;

// sqrt(3.0*dp*dp) support of the kernel
const real_number H = 0.00736121593217;

// Eta in the formulas
const real_number Eta2 = 0.01 * H*H;

const real_number FourH2 = 4.0 * H*H;

// alpha in the formula
const real_number visco = 0.1;

// cbar in the formula (calculated later)
real_number cbar = 0.0;

// Mass of the fluid particles
const real_number MassFluid = 0.0000767656;

// Mass of the boundary particles
const real_number MassBound = 0.0000767656;

//

// End simulation time
#ifdef TEST_RUN
const real_number t_end = 0.004;
#else
const real_number t_end = 1.5;
#endif

// Gravity acceleration
const real_number gravity = 9.81;

// Reference densitu 1000Kg/m^3
const real_number rho_zero = 1000.0;

// Filled later require h_swl, it is b in the formulas
real_number B = 0.0;

// Constant used to define time integration
const real_number CFLnumber = 0.2;

// Minimum T
const real_number DtMin = 0.00001;

// Minimum Rho allowed
const real_number RhoMin = 700.0;

// Maximum Rho allowed
const real_number RhoMax = 1300.0;

// Filled in initialization
real_number max_fluid_height = 0.0;

// Properties

// FLUID or BOUNDARY
const size_t type = 0;

// Density
const int rho = 1;

// Density at step n-1
const int rho_prev = 2;

// Pressure
const int Pressure = 3;

// Delta rho calculated in the force calculation
const int drho = 4;

// calculated force
const int force = 5;

// velocity
const int velocity = 6;

// velocity at previous step
const int velocity_prev = 7;

const int red = 8;

const int red2 = 9;

// Type of the vector containing particles
typedef vector_dist_gpu<3,real_number,aggregate<unsigned int,real_number,  real_number,    real_number,     real_number,     real_number[3], real_number[3], real_number[3], real_number, real_number>> particles;
//                                              |          |             |               |                |                |               |               |               |            |
//                                              |          |             |               |                |                |               |               |               |            |
//                                             type      density       density        Pressure          delta            force          velocity        velocity        reduction     another
//                                                                     at n-1                           density                                         at n - 1        buffer        reduction buffer


struct ModelCustom
{
	template<typename Decomposition, typename vector> inline void addComputation(Decomposition & dec,
			                                                                     vector & vd,
																				 size_t v,
																				 size_t p)
	{
		if (vd.template getProp<type>(p) == FLUID)
			dec.addComputationCost(v,4);
		else
			dec.addComputationCost(v,3);
	}

	template<typename Decomposition> inline void applyModel(Decomposition & dec, size_t v)
	{
		dec.setSubSubDomainComputationCost(v, dec.getSubSubDomainComputationCost(v) * dec.getSubSubDomainComputationCost(v));
	}

	real_number distributionTol()
	{
		return 1.01;
	}
};

template<typename vd_type>
__global__ void mark(vd_type vd)
{
	auto a = GET_PARTICLE(vd);

	real_number rho_a = vd.template getProp<rho>(a);
	real_number rho_frac = rho_a / rho_zero;

	vd.template getProp<red>(a) = a % 2;
}

size_t cnt = 0;


int main(int argc, char* argv[])
{
    // OpenFPM GPU distribution

    // OpenFPM by default select GPU 0 for process 0, gpu 1 for process 1 and so on ... . In case of multi-node is the same each node has
    // has a group of processes and these group of processes are distributed across the available GPU on that node.

    // If you want to override this behaviour use #define EXTERNAL_SET_GPU at the very beginning of the program and use
    // cudaSetDevice to select the GPU for that particular process before openfpm_init
    // Note: To get the process number do MPI_Init and and use the MPI_Comm_rank. VCluster is not available before openfpm_init
    // A code snippet in case we want to skip GPU 0
    // MPI_Init(&argc,&argv);
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    // cudaSetDevice(1+rank);

    // initialize the library
	openfpm_init(&argc,&argv);

	openfpm::vector<double> tr;

	for (int i = 0 ; i < 105 ; i++)
	{

#if !defined(CUDA_ON_CPU) && !defined(__HIP__)
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

	// It contain for each time-step the value detected by the probes
	openfpm::vector<openfpm::vector<real_number>> press_t;
	openfpm::vector<Point<3,real_number>> probes;

	probes.add({0.8779f,0.3f,0.02f});
	probes.add({0.754f,0.31f,0.02f});

	// Here we define our domain a 2D box with internals from 0 to 1.0 for x and y
	Box<3,real_number> domain({-0.05f,-0.05f,-0.05f},{1.7010f,0.7065f,0.511f});
	size_t sz[3] = {826,358,266};

	// Here we define the boundary conditions of our problem
    size_t bc[3]={NON_PERIODIC,NON_PERIODIC,NON_PERIODIC};

	// extended boundary around the domain, and the processor domain
	Ghost<3,real_number> g(2*H);

	particles vd(0,domain,bc,g,DEC_GRAN(128));

	//! \cond [draw fluid] \endcond

	// You can ignore all these dp/2.0 is a trick to reach the same initialization
	// of Dual-SPH that use a different criteria to draw particles
	Box<3,real_number> fluid_box({dp/2.0f,dp/2.0f,dp/2.0f},{0.4f+dp/2.0f,0.67f-dp/2.0f,0.3f+dp/2.0f});

	// return an iterator to the fluid particles to add to vd
	auto fluid_it = DrawParticles::DrawBox(vd,sz,domain,fluid_box);

	// here we fill some of the constants needed by the simulation
	max_fluid_height = fluid_it.getBoxMargins().getHigh(2);
	h_swl = fluid_it.getBoxMargins().getHigh(2) - fluid_it.getBoxMargins().getLow(2);
	B = (coeff_sound)*(coeff_sound)*gravity*h_swl*rho_zero / gamma_;
	cbar = coeff_sound * sqrt(gravity * h_swl);

	// for each particle inside the fluid box ...
	while (fluid_it.isNext())
	{
		// ... add a particle ...
		vd.add();

		// ... and set it position ...
		vd.getLastPos()[0] = fluid_it.get().get(0);
		vd.getLastPos()[1] = fluid_it.get().get(1);
		vd.getLastPos()[2] = fluid_it.get().get(2);

		// and its type.
		vd.template getLastProp<type>() = FLUID;

		// We also initialize the density of the particle and the hydro-static pressure given by
		//
		// rho_zero*g*h = P
		//
		// rho_p = (P/B + 1)^(1/Gamma) * rho_zero
		//

		vd.template getLastProp<Pressure>() = rho_zero * gravity *  (max_fluid_height - fluid_it.get().get(2));

		vd.template getLastProp<rho>() = pow(vd.template getLastProp<Pressure>() / B + 1, 1.0/gamma_) * rho_zero;
		vd.template getLastProp<rho_prev>() = vd.template getLastProp<rho>();
		vd.template getLastProp<velocity>()[0] = 0.0;
		vd.template getLastProp<velocity>()[1] = 0.0;
		vd.template getLastProp<velocity>()[2] = 0.0;

		vd.template getLastProp<velocity_prev>()[0] = 0.0;
		vd.template getLastProp<velocity_prev>()[1] = 0.0;
		vd.template getLastProp<velocity_prev>()[2] = 0.0;

		// next fluid particle
		++fluid_it;
	}

	// Recipient
	Box<3,real_number> recipient1({0.0f,0.0f,0.0f},{1.6f+dp/2.0f,0.67f+dp/2.0f,0.4f+dp/2.0f});
	Box<3,real_number> recipient2({dp,dp,dp},{1.6f-dp/2.0f,0.67f-dp/2.0f,0.4f+dp/2.0f});

	Box<3,real_number> obstacle1({0.9f,0.24f-dp/2.0f,0.0f},{1.02f+dp/2.0f,0.36f,0.45f+dp/2.0f});
	Box<3,real_number> obstacle2({0.9f+dp,0.24f+dp/2.0f,0.0f},{1.02f-dp/2.0f,0.36f-dp,0.45f-dp/2.0f});
	Box<3,real_number> obstacle3({0.9f+dp,0.24f,0.0f},{1.02f,0.36f,0.45f});

	openfpm::vector<Box<3,real_number>> holes;
	holes.add(recipient2);
	holes.add(obstacle1);
	auto bound_box = DrawParticles::DrawSkin(vd,sz,domain,holes,recipient1);

	while (bound_box.isNext())
	{
		vd.add();

		vd.getLastPos()[0] = bound_box.get().get(0);
		vd.getLastPos()[1] = bound_box.get().get(1);
		vd.getLastPos()[2] = bound_box.get().get(2);

		vd.template getLastProp<type>() = BOUNDARY;
		vd.template getLastProp<rho>() = rho_zero;
		vd.template getLastProp<rho_prev>() = rho_zero;
		vd.template getLastProp<velocity>()[0] = 0.0;
		vd.template getLastProp<velocity>()[1] = 0.0;
		vd.template getLastProp<velocity>()[2] = 0.0;

		vd.template getLastProp<velocity_prev>()[0] = 0.0;
		vd.template getLastProp<velocity_prev>()[1] = 0.0;
		vd.template getLastProp<velocity_prev>()[2] = 0.0;

		++bound_box;
	}

	auto obstacle_box = DrawParticles::DrawSkin(vd,sz,domain,obstacle2,obstacle1);

	while (obstacle_box.isNext())
	{
		vd.add();

		vd.getLastPos()[0] = obstacle_box.get().get(0);
		vd.getLastPos()[1] = obstacle_box.get().get(1);
		vd.getLastPos()[2] = obstacle_box.get().get(2);

		vd.template getLastProp<type>() = BOUNDARY;
		vd.template getLastProp<rho>() = rho_zero;
		vd.template getLastProp<rho_prev>() = rho_zero;
		vd.template getLastProp<velocity>()[0] = 0.0;
		vd.template getLastProp<velocity>()[1] = 0.0;
		vd.template getLastProp<velocity>()[2] = 0.0;

		vd.template getLastProp<velocity_prev>()[0] = 0.0;
		vd.template getLastProp<velocity_prev>()[1] = 0.0;
		vd.template getLastProp<velocity_prev>()[2] = 0.0;

		++obstacle_box;
	}

	vd.map();

	// Now that we fill the vector with particles
	ModelCustom md;

	vd.addComputationCosts(md);
	vd.getDecomposition().decompose();
	vd.map();

	///////////////////////////

	// Ok the initialization is done on CPU on GPU we are doing the main loop, so first we offload all properties on GPU

	vd.hostToDevicePos();
	vd.template hostToDeviceProp<type,rho,rho_prev,Pressure,velocity,velocity_prev>();

	vd.ghost_get<type,rho,Pressure,velocity>(RUN_ON_DEVICE);

	/////////////////////////////

	// remove the particles marked

	auto & v_cl = create_vcluster();

	timer tm;

	tm.start();
	openfpm::vector<size_t> rm;
	rm.reserve(vd.size_local() / 2);

	for (int i = 0 ; i < vd.size_local() / 2; i++)
	{
		rm.add(i * 2);
	}

        vd.remove(rm);

	tm.stop();

	if (v_cl.rank() == 0)
	{std::cout << "REMOVE: " << tm.getwct() << std::endl;}

	}

	auto & v_cl = create_vcluster();

	double rem_mean;
	double rem_dev;
	standard_deviation(tr,rem_mean,rem_dev);

	// mean across processors
	v_cl.max(rem_mean);
	v_cl.execute();

	if (v_cl.rank() == 0)
	{std::cout << "REM: " << rem_mean << std::endl;}

	openfpm_finalize();
}
 
#else

int main(int argc, char* argv[])
{
        return 0;
}

#endif
