cmake_minimum_required(VERSION 4.3)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNet
        src/matrix.cpp
        src/MNIST.cpp
        src/layer.cpp
        src/network.cpp
)

target_include_directories(NeuralNet PUBLIC include)