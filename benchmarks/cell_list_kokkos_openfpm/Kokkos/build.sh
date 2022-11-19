#! /bin/bash

source $HOME/openfpm_vars_4.1.0

cmake . -DBOOST_ROOT=/home/i-bird/openfpm_dependencies/BOOST/ -DKokkos_DIR=/home/i-bird/Desktop/MOSAIC/OpenFPM_project/kokkos_install/lib64/cmake/Kokkos/ -Dopenfpm_DIR=/usr/local/openfpm_4.1.0_openmp/cmake/

