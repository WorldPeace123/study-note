## FFTW3

```shell
wget http://www.fftw.org/fftw-3.3.8.tar.gz
tar xzvf fftw-3.3.8.tar.gz
cd ./fftw-3.3.8
./configure --enable-shared
make
sudo make install
sudo ldconfig
```



## AMD-ROCM安装

[Ubuntu native installation — ROCm installation (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/native-install/ubuntu.html)

```shell
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    
echo "deb [arch=amd64,i386 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.3.1/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list
    
sudo apt update

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.1 jammy main" \
    | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update

sudo apt install amdgpu-dkms
sudo reboot

sudo apt install rocm

##Let me see see
rocm-smi
which hipcc
sudo rocminfo
```

## Anaconda安装

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
chmod +x Anaconda3-2023.07-2-Linux-x86_64.sh
bash Anaconda3-2023.07-2-Linux-x86_64.sh
conda create -n allegro python==3.9
conda activate allegro
```



## pytorch with ROCM

```bash
##建议提前装好OneAPI套件，至少有mkl库

sudo rocminfo | grep gfx
#  Name:                    gfx1032                            
#  Name:                    amdgcn-amd-amdhsa--gfx1032  

git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive

set -e
SCRIPT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_PATH=${SCRIPT_DIR}/build
export ROCM_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/hip/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive
git checkout -- .
git clean -df -e build.sh

python3 tools/amd_build/build_amd.py

# 编译选项
# AMD Radeon 6600
export PYTORCH_ROCM_ARCH="gfx1032"  
export _GLIBCXX_USE_CXX11_ABI=1
export USE_NUMPY=1
export USE_ROCM=1
export USE_LMDB=1
export USE_OPENCV=1
export USE_CUDA=0
export USE_NINJA=0
export BUILD_CAFFE2=0
export BUILD_CAFFE2_OPS=0
export BUILD_TEST=0
export MAX_JOBS=32
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# 编译 
python3 setup.py build
#error python3 setup.py clean
python3 setup.py install
```

