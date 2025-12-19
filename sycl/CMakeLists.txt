# This file is part of acg.
#
# Copyright 2025 Koç University and Simula Research Laboratory
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors:
#  James D. Trotter <james@simula.no>
#  Sinan Ekmekçibaşı <sekmekcibasi23@ku.edu.tr>
#
# Last modified: 2025-04-26

cmake_minimum_required(VERSION 3.12)

project(acg-sycl LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

set(CMAKE_CXX_STANDARD 17)

set(PROJECT_SOURCES
	../acg/config.h
	../acg/cg.c
	../acg/cg.h
	../acg/cgsycl.cpp
	../acg/cgsycl.h
	../acg/cg-kernels-sycl.cpp
	../acg/cg-kernels-sycl.h
	../acg/cgpetsc.c
	../acg/cgpetsc.h
	../acg/comm.c
	../acg/comm.h
	../acg/error.c
	../acg/error.h
	../acg/fmtspec.c
	../acg/fmtspec.h
	../acg/graph.c
	../acg/graph.h
	../acg/halo.c
	../acg/halo.h
	../acg/metis.c
	../acg/metis.h
	../acg/mtxfile.c
	../acg/mtxfile.h
	../acg/prefixsum.c
	../acg/prefixsum.h
	../acg/sort.c
	../acg/sort.h
	../acg/symcsrmatrix.c
	../acg/symcsrmatrix.h
	../acg/vector.c
	../acg/vector.h
	acg-sycl.cpp
	)


  include(CheckCSourceCompiles)
check_c_source_compiles(
"#define _POSIX_C_SOURCE 200112L
#include <time.h>
int main () {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return 0;
}" HAVE_CLOCK_GETTIME)
if(HAVE_CLOCK_GETTIME)
    add_definitions(-DHAVE_CLOCK_GETTIME)
endif()

find_package(OpenMP)
if (OpenMP_C_FOUND)
    add_definitions(-DACG_HAVE_OPENMP)
    message(STATUS "OpenMP is enabled")
endif ()

find_package(MPI REQUIRED)
if (MPI_FOUND)
    add_definitions(-DACG_HAVE_MPI)
endif ()

find_package(METIS)
if (METIS_FOUND)
  add_definitions(-DACG_HAVE_METIS)
  INCLUDE(CheckTypeSize)
  SET(CMAKE_REQUIRED_INCLUDES ${METIS_INCLUDE_DIRS})
  SET(CMAKE_EXTRA_INCLUDE_FILES "metis.h")
  CHECK_TYPE_SIZE("idx_t" SIZEOF_IDX_T)
  SET(CMAKE_REQUIRED_INCLUDES)
  SET(CMAKE_EXTRA_INCLUDE_FILES)
endif ()

# Find SYCL
if(NOT SYCL_FLAGS)
    # -fsycl is the magic flag for dpcpp/clang++ to enable SYCL
    set(SYCL_FLAGS "-fsycl")
endif()

add_definitions(-DACG_HAVE_SYCL)



find_package(ZLIB)
if (ZLIB_FOUND)
    add_definitions(-DACG_HAVE_LIBZ)
endif ()

if (IDXSIZE EQUAL 64)
  if (NOT SIZEOF_IDX_T EQUAL 8)
    message(SEND_ERROR "failed to detect METIS with support for 64-bit indices (sizeof(idx_t)=${SIZEOF_IDX_T}) - this may lead to incorrect results when setting ACG_IDX_SIZE=64")
  endif ()
  add_definitions(-DACG_IDX_SIZE=64)
endif ()

if (ACG_ENABLE_PROFILING)
  add_definitions(-DACG_ENABLE_PROFILING)
  message(STATUS "Profiling enabled")
endif ()

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC IMPORTED_TARGET petsc)
if (PETSC_FOUND)
  add_definitions(-DACG_HAVE_PETSC)
  INCLUDE(CheckTypeSize)
  SET(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
  SET(CMAKE_EXTRA_INCLUDE_FILES "petsc.h")
  CHECK_TYPE_SIZE("PetscInt" SIZEOF_PETSC_INT)
  SET(CMAKE_REQUIRED_INCLUDES)
  SET(CMAKE_EXTRA_INCLUDE_FILES)
  if (IDXSIZE EQUAL 64)
    if (NOT SIZEOF_PETSC_INT EQUAL 8)
      message(SEND_ERROR "failed to detect PETSc with support for 64-bit integers (sizeof(PetscInt)=${SIZEOF_PETSC_INT}) - this may lead to incorrect results when setting ACG_IDX_SIZE=64")
    endif ()
  endif ()
endif ()

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# 1. Apply SYCL flags ONLY to the C++ files to prevent 
# C files from being compiled as C++ (fixing the malloc errors)
set(SYCL_SOURCE_FILES
    "../acg/cgsycl.cpp"
    "../acg/cg-kernels-sycl.cpp"
    "acg-sycl.cpp"
)
set_source_files_properties(${SYCL_SOURCE_FILES} PROPERTIES COMPILE_FLAGS "${SYCL_FLAGS}")

# 2. IMPORTANT: Apply SYCL flags to the LINKER step
# This tells clang++ to link against libsycl.so and handle NVIDIA device code
target_link_options(${PROJECT_NAME} PRIVATE ${SYCL_FLAGS})


target_compile_options(${PROJECT_NAME} PRIVATE -Wno-reserved-user-defined-literal)


target_link_libraries(${PROJECT_NAME} MPI::MPI_C m)
target_include_directories(${PROJECT_NAME} PRIVATE ..)

if(OpenMP_C_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_C)
endif()

if (METIS_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${METIS_INCLUDE_DIRS})
  target_link_directories(${PROJECT_NAME} PRIVATE ${METIS_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} ${METIS_LIBRARY})
endif ()
if (PETSC_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${PETSC_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PkgConfig::PETSC)
endif ()

if (ZLIB_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${ZLIB_INCLUDE_DIRS})
  target_link_directories(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} z)
endif ()
