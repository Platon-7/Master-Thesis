CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(rela)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -Wfatal-errors")

execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/get_pybind_flags.py
  OUTPUT_VARIABLE PYBIND_FLAGS
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS}")
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

# find_package(PythonInterp 3.7 REQUIRED)
# find_package(PythonLibs 3.7 REQUIRED)

# find pytorch
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

# Manual libtorch linkage — deliberately NOT using find_package(Torch).
# find_package(Torch) pulls in torch's Caffe2 CUDA cmake, which fails on this
# cluster: the env's torch is built for CUDA 12.1 but the system CUDA toolkit is
# 13.0 and not reliably present on every node. It hunts for CUDA-12-era files
# that CUDA 13 removed (nvToolsExt / libnvToolsExt) and for a full toolkit
# (nvcc, cudart, headers) we don't need. `rela` has NO CUDA sources of its own,
# so it only needs libtorch's headers + libs to build; any CUDA tensor ops
# resolve at RUNTIME via libtorch's bundled CUDA libraries. So set the paths by
# hand and skip the toolkit machinery entirely.
set(TORCH_INCLUDE_DIRS
  "${TorchPath}/include"
  "${TorchPath}/include/torch/csrc/api/include")
file(GLOB TORCH_LIBRARIES
  "${TorchPath}/lib/libtorch.so"
  "${TorchPath}/lib/libtorch_cpu.so"
  "${TorchPath}/lib/libc10.so")
set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")

message("---------------------")
message(${TorchPath})
message(${PYBIND_FLAGS})
message(${CMAKE_CXX_FLAGS})
message("---------------------")


# lib for other rela programs
add_library(rela_lib
  rela/transition.cc
  rela/replay.cc
  rela/concurrent_queue.cc
  rela/tensor_dict.cc
  rela/episode.cc
)
target_include_directories(rela_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rela_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela_lib PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

pybind11_add_module(rela rela/pybind.cc)
target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
