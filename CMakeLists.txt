cmake_minimum_required(VERSION 3.20)

project(CudaTriangleCounting LANGUAGES CXX CUDA)

# Gunrock-style defaults: explicit CUDA arch and release build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA SM architectures (e.g. 86;89;90)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_ITERATOR_MODES "Build iterator-based modes" ON)
option(BUILD_TENSOR_CORE_MODES "Build tensor-core modes" ON)

set(COMMON_SOURCES
  ${PROJECT_SOURCE_DIR}/Libs/GraphReader.cu
  ${PROJECT_SOURCE_DIR}/Libs/ChronoCuda.cu
  ${PROJECT_SOURCE_DIR}/Libs/CudaUtilities.cu
  ${PROJECT_SOURCE_DIR}/Libs/GraphStats.cu
  ${PROJECT_SOURCE_DIR}/Libs/TileReader.cu
  
)

set(TENSOR_COMMON_SOURCES
  ${PROJECT_SOURCE_DIR}/Libs/TensorUtilities.cu
)

function(add_triangle_target target_name main_source)
  set(options WITH_TENSOR_UTILS)
  cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})

  set(target_sources ${main_source} ${COMMON_SOURCES})
  if(ARG_WITH_TENSOR_UTILS)
    list(APPEND target_sources ${TENSOR_COMMON_SOURCES})
  endif()

  add_executable(${target_name} ${target_sources})

  target_include_directories(${target_name}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/Libs
  )

  target_compile_options(${target_name}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  )
endfunction()

if(BUILD_ITERATOR_MODES)
  add_triangle_target(e "${PROJECT_SOURCE_DIR}/Mode/Iterator/Edge/E.cu")
  add_triangle_target(ef "${PROJECT_SOURCE_DIR}/Mode/Iterator/EdgeForward/EF.cu")
  add_triangle_target(ew "${PROJECT_SOURCE_DIR}/Mode/Iterator/EdgeWarped/EW.cu")
  add_triangle_target(n "${PROJECT_SOURCE_DIR}/Mode/Iterator/Node/N.cu")
endif()

if(BUILD_TENSOR_CORE_MODES)
  add_triangle_target(ttc_bft "${PROJECT_SOURCE_DIR}/Mode/Tensor Core/Block Full Tile/TTC_BFT.cu" WITH_TENSOR_UTILS)
  add_triangle_target(ttc_bhfo "${PROJECT_SOURCE_DIR}/Mode/Tensor Core/Block Hadamard Full Operation/TTC_BHFO.cu" WITH_TENSOR_UTILS)
  add_triangle_target(ttc_bhso "${PROJECT_SOURCE_DIR}/Mode/Tensor Core/Block Hadamard Sparse Operation/TTC_BHSO.cu" WITH_TENSOR_UTILS)
  add_triangle_target(stconv "${PROJECT_SOURCE_DIR}/Mode/Tensor Core/SparseTileReader/converter/ST_converter.cu" WITH_TENSOR_UTILS)
  add_triangle_target(stcomp "${PROJECT_SOURCE_DIR}/Mode/Tensor Core/SparseTileReader/compute/ST_compute.cu" WITH_TENSOR_UTILS)
  set_target_properties(ttc_bft ttc_bhfo ttc_bhso stconv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
