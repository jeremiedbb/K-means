wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
rm ~/miniconda.sh

export PATH=${HOME}/miniconda/bin:${PATH}

conda install -y numpy
conda install -y cython
conda install -y scipy

git clone https://github.com/scikit-learn/scikit-learn.git ~/scikit-learn
apt-get install --assume-yes gcc
apt-get install --assume-yes g++
apt-get install --assume-yes make
make -C ~/scikit-learn clean inplace

export PYTHONPATH=${HOME}/scikit-learn:${PYTHONPATH}

conda update -y conda
conda config --add channels intel
conda create -y -n intel_python intelpython3_core python=3

source activate intel_python
conda install -y scikit-learn
source deactivate

apt-get install htop