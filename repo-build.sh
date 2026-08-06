#!/bin/bash
set -e

# cmake -B build \
#   -DGGML_METAL=ON \
#   -DCMAKE_C_FLAGS="-I$(brew --prefix libomp)/include" \
#   -DCMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include" \
#   -DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix libomp)/lib -lomp"

cmake -B build -DGGML_METAL=ON -DGGML_OPENMP=OFF

cmake --build build --config Release -j

cd tools/ui
npm install
npm run build
