# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX)

# You need this for using Kokkos_ROOT variable
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12.0")
  message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(OpenMP)
find_package(Threads)
find_package(MPI)
find_package(Boost 1.75.0 COMPONENTS program_options iostreams filesystem fiber context REQUIRED)

# Look for an installed Kokkos
find_package(openfpm 4.1.0 REQUIRED)
find_package(Kokkos REQUIRED)

add_definitions( -D__NVCC__ -DCUDART_VERSION=11000 )

add_executable(cell_list cell_list_interation.cpp)

# This is the only thing required to set up compiler/linker flags
target_link_libraries(cell_list Kokkos::kokkos)
target_link_libraries(cell_list  openfpm::binary_config)
