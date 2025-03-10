cmake_minimum_required(VERSION 3.10)
project(ViTEmbeddingExtractor)

# ✅ Force C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ✅ Set the correct compiler
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

# ✅ Set LibTorch path
set(CMAKE_PREFIX_PATH "/mnt/c/Users/hcang/OneDrive/Desktop/torch_cpp_test/libtorch")

find_package(Torch REQUIRED)

add_executable(vit_embedding main.cpp)
target_link_libraries(vit_embedding "${TORCH_LIBRARIES}")
