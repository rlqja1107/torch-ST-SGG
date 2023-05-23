# Installation

You should install the necessary package to run the code. For using the recent pytorch version, we use the pytorch version as 3.9.0 with CUDA 11.3 version. 

## Versions of main package 
- Python == 3.9.0

- PyTorch == 1.10.1 (CUDA: 11.3)

- torchvision == 0.11.2

- numpy == 1.24.3

- yacs == 0.1.8



## Guide for installation

```bash

# Note that python version is 3.9.0
conda create --n stsgg python==3.9.0
conda activate stsgg

conda install -y ipython scipy h5py

pip install ninja yacs cython matplotlib tqdm opencv-python overrides gpustat gitpython ipdb graphviz tensorboardx termcolor scikit-learn==1.2.2

# For the recent version, we use the pytorch 1.10.1 version with CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install torch-spline-conv torch-cluster -f https://data.pyg.org/whl/torch-1.10.1+11.3.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+11.3.html
pip install torch-geometric

# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd ../..
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

cd ..
python setup.py build develop


```