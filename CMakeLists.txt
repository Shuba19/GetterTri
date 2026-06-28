cmake_minimum_required(VERSION 3.20)

project(CudaTriangleCounting LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

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

set(COMMON_SOURCES
  ${PROJECT_SOURCE_DIR}/Libs/GraphReader.cu
  ${PROJECT_SOURCE_DIR}/Libs/ChronoCuda.cu
  ${PROJECT_SOURCE_DIR}/Libs/CudaUtilities.cu
  ${PROJECT_SOURCE_DIR}/Libs/GraphStats.cu
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

  target_link_libraries(${target_name}
    PRIVATE
      OpenMP::OpenMP_CXX
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
  add_triangle_target(ew "${PROJECT_SOURCE_DIR}/Mode/Iterator/EdgeWarped/EW.cu")
  add_triangle_target(n "${PROJECT_SOURCE_DIR}/Mode/Iterator/Node/N.cu")
endif()
