# vit embedding extractor

this project demonstrates how to export a vision transformer (vit) model from pytorch/python and run inference using libtorch c++.

## project structure

- `main.py` - python script to export the vit model to torchscript format
- `inference/` - c++ inference implementation
  - `main.cpp` - c++ code for running inference
  - `cmakelists.txt` - cmake configuration file
  - `model_location/` - directory containing the exported model (you need to move this to the build folder once the compilation is successful)

## prerequisites

- python requirements:
  - pytorch
  - transformers library (`transformers`)  // for conversion
- c++ requirements:
  - cmake (>= 3.10)
  - gcc-11
  - libtorch (cpu version) // or do gpu, suit yourself

## setup

1. install libtorch:

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

2. export the vit model:

```bash
python main.py
```

3. build the c++ inference code:

```bash
cd inference
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_PREFIX_PATH={libtorch_location} ..
cmake --build . --config Release
```

## running inference

after building, run the inference executable:

```bash
./vit_embedding
```

the program will:

1. load the exported vit model
2. create a random test input tensor (1x3x224x224)
3. run inference to extract embeddings
4. display the resulting embedding tensor

## model details

this project uses the `google/vit-base-patch16-224` vision transformer model to extract embeddings. the model:

- takes input images of size 224x224 pixels
- processes them in patches of 16x16
- outputs an embedding vector from the [cls] token

## license

this project uses the vit model from the hugging face transformers library. please refer to their licensing terms for model usage.
