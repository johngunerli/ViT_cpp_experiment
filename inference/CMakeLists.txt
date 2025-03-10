cmake_minimum_required(VERSION 3.10)
project(ViTEmbeddingExtractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


# Specify the LibTorch path
set(CMAKE_PREFIX_PATH "/Users/hakancangunerli/Desktop/onnx_er/libtorch")

find_package(Torch REQUIRED)

add_executable(vit_embedding main.cpp)
target_link_libraries(vit_embedding "${TORCH_LIBRARIES}")
set_property(TARGET vit_embedding PROPERTY CXX_STANDARD 14)
