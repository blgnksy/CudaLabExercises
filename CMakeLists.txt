cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

if (CUDA_FOUND)
  message("CUDA found, using device squaring!")
  #add_subdirectory(src.gpu)
else()
  message("CUDA not found, doing something alternatively")
  #add_subdirectory(src.cpu)
endif()

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -gencode arch=compute_61,code=sm_61
)

cuda_add_executable(
matrix_matrix_mul
matrix_matrix_mul.cu)
