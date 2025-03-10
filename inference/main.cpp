#include <torch/script.h>
#include <iostream>

// install libtorch from 
// wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
// unzip libtorch-shared-with-deps-latest.zip

int main() {

    // print the location of where this code is 

    std::cout << "Current Path: " << __FILE__ << std::endl;

    torch::jit::script::Module model;
    model = torch::jit::load("vit_embedding.pt");
    std::cout << "Model loaded successfully!" << std::endl;

    // Create an input tensor (dummy example, will replace with actual image, or make it dynamic per input)
    torch::Tensor input_tensor = torch::rand({1, 3, 224, 224});

    // Run inference
    torch::Tensor embedding = model.forward({input_tensor}).toTensor();

    // std::cout << "Embedding Shape: " << embedding.sizes() << std::endl;
    
    // show the reslting emebdding 

    std::cout << embedding << std::endl;

    return 0;
}
