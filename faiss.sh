#!/bin/bash
set -e  # Exit on error
set -o pipefail  # Catch pipe errors

# --------------------------
# 0. Add Debug Output
# --------------------------
echo "=== Starting FAISS GPU Installation ==="

# --------------------------
# 1. Install System Dependencies (with progress)
# --------------------------
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    cmake \
    git \
    libopenblas-dev \
    libomp-dev \
    python3-dev \
    python3-pip \
    swig \

# --------------------------
# 2. Install CMake (with verbose output)
# --------------------------
echo "[2/6] Downloading CMake..."
wget -v https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz

echo "[3/6] Building CMake (this takes 5-15 mins)..."
tar -xzf cmake-3.29.3.tar.gz
cd cmake-3.29.3
./bootstrap --system-curl --parallel=$(nproc) | tee -a ../cmake_build.log
make -j$(($(nproc) - 1)) VERBOSE=1 | tee -a ../cmake_build.log
sudo make install | tee -a ../cmake_build.log
cd ..

# --------------------------
# 3. FAISS Build (with visible progress)
# --------------------------
echo "[4/6] Cloning FAISS repository..."
git clone https://github.com/facebookresearch/faiss.git
cd faiss

echo "[5/6] Configuring FAISS build..."
cmake -B build \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_CUDA_ARCHITECTURES="75;80" \
    -DCUDAToolkit_ROOT=/usr/local/cuda 2>&1 | tee ../faiss_config.log

echo "[6/6] Building FAISS (this takes 10-30 mins)..."
make -C build -j$(($(nproc) - 2)) VERBOSE=1 2>&1 | tee ../faiss_build.log

# --------------------------
# 4. Final Installation
# --------------------------
echo "=== Final Installation Steps ==="
sudo make -C build install
cd python
pip install . 2>&1 | tee ../python_install.log

echo "=== Installation Complete ==="
