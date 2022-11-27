#include <Vector/map_vector.hpp>
#include <chrono>
//#include <memory/HeapMemory.cpp>


/////////////////////////////////////////////// Layouts for fun //////////////////////////////////////////////////

// Each <> is separated buffer, now the leading dimension in case of vectorial or tensorial is [], not anymore the () parenthesis

//! Partial specialization for scalar N=0
template<typename T>
struct t_to_memory_c_custom_impl
{
	//! transform a type T into memory_c<T>
	typedef memory_c<T> type;
};


//! Partial specialization for N=1
template<typename T,size_t N1>
struct t_to_memory_c_custom_impl<T[N1]>
{
	//! transfrom a type T into memory_c<T>
	typedef memory_c<Point<N1,T>> type;

	//! the internal array primitive information represented into a boost mpl vector
	typedef boost::mpl::vector<T,boost::mpl::int_<N1>> prim_vmpl;
};


template<typename T>
struct t_to_memory_c_custom
{
	typedef typename t_to_memory_c_custom_impl<typename remove_attributes_const_ref<T>::type>::type type;
};


template<typename Taggr>
struct memory_traits_custom_split_buff : public memory_traits_inte<Taggr>
{
	typedef int yes_is_inte;

	typedef typename v_transform<t_to_memory_c_custom,typename Taggr::type>::type type;

};

template<unsigned int dim, typename T, typename S, typename linearizer  >
class grid_base<dim, T,S, typename memory_traits_custom_split_buff<T>::type, linearizer >: public grid_base_impl<dim,T,S,memory_traits_custom_split_buff,linearizer>
{
	public:

	inline grid_base(const size_t & sz) THROW
	:grid_base_impl<dim,T,S,memory_traits_custom_split_buff,linearizer>(sz)
	{
	}
};

template<unsigned int dim,typename T>
class encapc<dim,T,typename memory_traits_custom_split_buff<T>::type>
{
	typedef typename memory_traits_custom_split_buff<T>::type type;

	//! reference to the encapsulated object
	type & data;

	//! element id
	size_t k;

	public:

	__device__ __host__ encapc(typename memory_traits_custom_split_buff<T>::type & data, size_t k)
	:data(data),k(k)
	{}

	template <unsigned int p>
	__device__ __host__ auto get() -> decltype(boost::fusion::at_c<p>(data).mem_r.operator[](k)) &
	{
		return boost::fusion::at_c<p>(data).mem_r.operator[](k);
	}

	template <unsigned int p> 
	__device__ __host__ auto get() const -> decltype(boost::fusion::at_c<p>(data).mem_r.operator[](k)) &
	{
		return boost::fusion::at_c<p>(data).mem_r.operator[](k);
	}
};

///////////////////////////////////////// AOS ///////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//#if defined __GNUC__                       // Got the idea but it seems incorrect for clang __GNUC__ is defined, in clang it end here
//#	define IVDEP _Pragma("GCC ivdep")
//#elif defined(_MSC_VER)
//#	define IVDEP __pragma(loop(ivdep))
//#elif defined __clang__
//#	define IVDEP _Pragma("clang loop vectorize(enable) interleave(enable) distribute(enable)")
#define IVDEP _Pragma("clang loop vectorize(assume_safety) interleave(assume_safety)")
//#else
//#	error "Please define IVDEP for your compiler!"
//#endif

#define FLAT_AGGREGATE 0

namespace {
	struct Stopwatch {
		using clock = std::chrono::high_resolution_clock;

		auto elapsedAndReset() -> double {
			const auto now = clock::now();
			const auto seconds = std::chrono::duration<double>{now - last}.count();
			last = now;
			return seconds;
		}

	private:
		clock::time_point last = clock::now();
	};

	constexpr auto PROBLEM_SIZE = 16 * 1024;
	constexpr auto STEPS = 5;
	constexpr auto TIMESTEP = 0.0001f;
	constexpr auto EPS2 = 0.01f;

	template <typename Data, template <typename> typename LayoutBase>
	using OpenFPMVector = openfpm::vector<Data, HeapMemory, LayoutBase>;

#if FLAT_AGGREGATE
	using Data = aggregate<float, float, float, float, float, float, float>;

        constexpr auto POS_X = 0;
        constexpr auto POS_Y = 1;
        constexpr auto POS_Z = 2;
        constexpr auto VEL_X = 3;
        constexpr auto VEL_Y = 4;
        constexpr auto VEL_Z = 5;
        constexpr auto MASS = 6;

#else
	using Data = aggregate<float[3], float[3], float>;

        constexpr auto POS = 0;
        constexpr auto VEL = 1;
        constexpr auto MASS = 2;

#endif

	//using Vector = OpenFPMVector<Data, memory_traits_lin>;
	//using Vector = OpenFPMVector<Data, memory_traits_inte>;
	using Vector = OpenFPMVector<Data,memory_traits_custom_split_buff>;


	inline void pPInteraction(Vector& particles, int i, int j) {
#if FLAT_AGGREGATE
		const float distanceX = particles.get<POS_X>(i) - particles.get<POS_X>(j);
		const float distanceY = particles.get<POS_Y>(i) - particles.get<POS_Y>(j);
		const float distanceZ = particles.get<POS_Z>(i) - particles.get<POS_Z>(j);
		const float distanceSqrX = distanceX * distanceX;
		const float distanceSqrY = distanceY * distanceY;
		const float distanceSqrZ = distanceZ * distanceZ;
		const float distSqr = EPS2 + distanceSqrX + distanceSqrY + distanceSqrZ;
		const float distSixth = distSqr * distSqr * distSqr;
		const float invDistCube = 1.0f / std::sqrt(distSixth);
		const float sts = particles.get<MASS>(j) * invDistCube * TIMESTEP;
		particles.get<VEL_X>(i) += distanceSqrX * sts;
		particles.get<VEL_Y>(i) += distanceSqrY * sts;
		particles.get<VEL_Z>(i) += distanceSqrZ * sts;
#else
		const auto pi = particles.get(i);
		const auto pj = particles.get(j);
		const auto posi = pi.get<POS>();
		const auto posj = pj.get<POS>();
		const float distanceX = posi[0] - posj[0];
		const float distanceY = posi[1] - posj[1];
		const float distanceZ = posi[2] - posj[2];
		const float distanceSqrX = distanceX * distanceX;
		const float distanceSqrY = distanceY * distanceY;
		const float distanceSqrZ = distanceZ * distanceZ;
		const float distSqr = EPS2 + distanceSqrX + distanceSqrY + distanceSqrZ;
		const float distSixth = distSqr * distSqr * distSqr;
		const float invDistCube = 1.0f / std::sqrt(distSixth);
		const float sts = pj.get<2>() * invDistCube * TIMESTEP;
		auto veli = pi.get<1>();
		veli[0] += distanceSqrX * sts;
		veli[1] += distanceSqrY * sts;
		veli[2] += distanceSqrZ * sts;

/////////////// Checking the proxies
//             	std::cout << demangle(typeid(decltype(pj)).name()) << std::endl;
//		std::cout << demangle(typeid(decltype(posj)).name()) << std::endl;

#endif
	}

	void update(Vector& particles) {
		for (int i = 0; i < PROBLEM_SIZE; i++)
			IVDEP
			for (int j = 0; j < PROBLEM_SIZE; j++)
				pPInteraction(particles, i, j);
	}

	void move(Vector& particles) {
		IVDEP
		for (std::size_t i = 0; i < PROBLEM_SIZE; i++) {
#if FLAT_AGGREGATE
			particles.get(i).get<POS_X>() += particles.get(i).get<VEL_X>() * TIMESTEP;
			particles.get(i).get<POS_Y>() += particles.get(i).get<VEL_Y>() * TIMESTEP;
			particles.get(i).get<POS_Z>() += particles.get(i).get<VEL_Z>() * TIMESTEP;
#else
			auto&& pi = particles.get(i);
			const auto& veli = pi.get<1>();
			auto&& posi = pi.get<0>();
			posi[0] += veli[0] * TIMESTEP;
			posi[1] += veli[1] * TIMESTEP;
			posi[2] += veli[2] * TIMESTEP;
#endif
		}
	}
}

int main() {
	Vector particles(PROBLEM_SIZE);

	{
		const auto& p0 = particles.get(0);
#if FLAT_AGGREGATE
		std::cout << "addresses:\n"
				  << &p0.get<POS_X>() << '\n'
				  << &p0.get<POS_Y>() << '\n'
				  << &p0.get<POS_Z>() << '\n'
				  << &p0.get<VEL_X>() << '\n'
				  << &p0.get<VEL_Y>() << '\n'
				  << &p0.get<VEL_Z>() << '\n'
				  << &p0.get<MASS>() << '\n';
#else
		std::cout << "addresses:\n"
				  << &p0.get<POS>()[0] << '\n'
				  << &p0.get<POS>()[1] << '\n'
				  << &p0.get<POS>()[2] << '\n'
				  << &p0.get<VEL>()[0] << '\n'
				  << &p0.get<VEL>()[1] << '\n'
				  << &p0.get<VEL>()[2] << '\n'
				  << &p0.get<MASS>() << '\n';
#endif
	}

	std::default_random_engine engine;
	std::normal_distribution<float> dist(float(0), float(1));
	for (auto i = 0; i < PROBLEM_SIZE; i++) {
#if FLAT_AGGREGATE
		particles.get(i).get<POS_X>() = dist(engine);
		particles.get(i).get<POS_Y>() = dist(engine);
		particles.get(i).get<POS_Z>() = dist(engine);
		particles.get(i).get<VEL_X>() = dist(engine) / float(10);
		particles.get(i).get<VEL_Y>() = dist(engine) / float(10);
		particles.get(i).get<VEL_Z>() = dist(engine) / float(10);
		particles.get(i).get<MASS>() = dist(engine) / float(100);
#else
		auto&& pi = particles.get(i);
		pi.get<POS>()[0] = dist(engine);
		pi.get<POS>()[1] = dist(engine);
		pi.get<POS>()[2] = dist(engine);
		pi.get<VEL>()[0] = dist(engine) / float(10);
		pi.get<VEL>()[1] = dist(engine) / float(10);
		pi.get<VEL>()[2] = dist(engine) / float(10);
		pi.get<MASS>() = dist(engine) / float(100);
#endif
	}

	Stopwatch watch;
	double sumUpdate = 0;
	double sumMove = 0;
	for (std::size_t s = 0; s < STEPS; ++s) {
		update(particles);
		sumUpdate += watch.elapsedAndReset();
		move(particles);
		sumMove += watch.elapsedAndReset();
	}
	std::cout << "openfpm\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

	{
#if FLAT_AGGREGATE
		const auto& p0 = particles.get(0);
		std::cout << "particle 0 pos: " << p0.get<0>() << " " << p0.get<1>() << " " << p0.get<2>() << '\n';
#else
		const auto& pos0 = particles.get<POS>(0);
		std::cout << "particle 0 pos: " << pos0[0] << " " << pos0[1] << " " << pos0[2] << '\n';
#endif
	}
}
