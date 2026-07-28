#!/bin/bash

# Adapted from https://github.com/mir-group/flare/blob/master/lammps_plugins/install.sh

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage:    ./install.sh path/to/lammps"
    exit 1
fi

lammps=$1

# add new lammps source files
ln -sf $(pwd)/pair_symmetrix_mace.h ${lammps}/src/pair_symmetrix_mace.h
ln -sf $(pwd)/pair_symmetrix_mace.cpp ${lammps}/src/pair_symmetrix_mace.cpp
ln -sf $(pwd)/pair_symmetrix_mace_kokkos.h ${lammps}/src/KOKKOS/pair_symmetrix_mace_kokkos.h
ln -sf $(pwd)/pair_symmetrix_mace_kokkos.cpp ${lammps}/src/KOKKOS/pair_symmetrix_mace_kokkos.cpp

# update lammps build instructions
echo "
function(symmetrix_add_lammps_library)
  # LAMMPS builds bundled Kokkos statically even when liblammps is shared.
  # Keep Symmetrix and KokkosKernels static so only one Kokkos runtime is linked.
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory($(pwd)/../libsymmetrix libsymmetrix)
endfunction()
symmetrix_add_lammps_library()
target_include_directories(lammps PRIVATE $(pwd)/../libsymmetrix/source)
target_link_libraries(lammps PRIVATE symmetrix)
" >> $lammps/cmake/CMakeLists.txt
