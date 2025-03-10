#include <torch/script.h>
#include <iostream>

// install libtorch from 
// wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
// unzip libtorch-shared-with-deps-latest.zip


int main() {
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("vit_embedding.pt");
        std::cout << "Model loaded successfully!" << std::endl;

        // Create an input tensor (dummy example, replace with actual image data)
        torch::Tensor input_tensor = torch::rand({1, 3, 224, 224});

        // Run inference
        torch::Tensor embedding = model.forward({input_tensor}).toTensor();

        std::cout << "Embedding Shape: " << embedding.sizes() << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
    }

    return 0;
}
