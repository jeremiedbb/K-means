wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
rm ~/miniconda.sh

export PATH=${HOME}/miniconda/bin:${PATH}

git clone https://github.com/scikit-learn/scikit-learn.git ~/scikit-learn
apt-get install --assume-yes gcc
apt-get install --assume-yes g++
apt-get install --assume-yes make

conda update -y conda
conda create -y -n dev python=3.6 numpy scipy cython
conda create -y -n intel -c intel intelpython3_core python=3

source activate dev
pip install -e ~/scikit-learn
source deactivate

source activate intel
conda install -y -c intel scikit-learn
source deactivate
