cmake_minimum_required(VERSION 3.23)
project(arco_entregable_2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(arco_entregable_2 main.cu)

set_target_properties(arco_entregable_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
