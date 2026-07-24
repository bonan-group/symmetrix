cmake_minimum_required(VERSION 3.20)
project(LibSymmetrix)

message(STATUS "Symmetrix: Beginning to process CMakeLists.txt.")

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----- dependencies -----

find_package(BLAS REQUIRED)
add_subdirectory(external/sphericart)
option(SYMMETRIX_KOKKOS "Enable Kokkos mode in Symmetrix." ON)
if(SYMMETRIX_KOKKOS)
    if(TARGET Kokkos::kokkos)
        message(STATUS "Symmetrix: Found existing Kokkos target.")
    else()
        add_subdirectory(external/kokkos)
    endif()
    add_subdirectory(external/kokkos-kernels)
endif()

# ----- libsymmetrix -----

set(SYMMETRIX_SOURCE_FILES
    source/affine_mlp.cpp
    source/e3nn.cpp
    source/e3nn_product.cpp
    source/model_metadata.cpp
    source/compact_radial.cpp
    source/cubic_spline.cpp
    source/cubic_spline_set.cpp
    source/mace.cpp
    source/mace_nonlinear.cpp
    source/mace_nonlinear_schema.cpp
    source/multilayer_perceptron.cpp
    source/multivariate_polynomial.cpp
    source/spherical_harmonic.cpp
    source/tools.cpp
    source/zbl.cpp)

if(SYMMETRIX_KOKKOS)
    list(APPEND SYMMETRIX_SOURCE_FILES
        source/affine_mlp_kokkos.cpp
        source/e3nn_kokkos.cpp
        source/e3nn_product_kokkos.cpp
        source/cubic_spline_kokkos.cpp
        source/cubic_spline_set_kokkos.cpp
        source/mace_kokkos.cpp
        source/mace_nonlinear_kokkos.cpp
        source/multilayer_perceptron_kokkos.cpp
        source/multivariate_polynomial_kokkos.cpp
        source/radial_function_set_kokkos.cpp
        source/tools_kokkos.cpp
        source/zbl_kokkos.cpp)
endif()

add_library(symmetrix ${SYMMETRIX_SOURCE_FILES})

# TODO: generalize
target_compile_options(symmetrix
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
if(NOT Kokkos_ENABLE_CUDA)
    target_compile_options(symmetrix PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

target_include_directories(symmetrix
    PUBLIC  source
            external/json/single_include
    PRIVATE external/sphericart/sphericart/include)

target_link_libraries(symmetrix
    PRIVATE ${BLAS_LIBRARIES}
            sphericart)
if(SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix
        PRIVATE Kokkos::kokkos
                Kokkos::kokkoskernels)
endif()


# ----- options -----

option(SYMMETRIX_SPHERICART_CUDA OFF)
if (SYMMETRIX_SPHERICART_CUDA)
    message(STATUS "Symmetrix: Will use CUDA version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_CUDA)
endif()
