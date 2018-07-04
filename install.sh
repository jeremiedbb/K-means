wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
rm ~/miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"

conda install -y numpy
conda install -y cython
conda install -y scipy

git clone https://github.com/scikit-learn/scikit-learn.git
apt-get install --assume-yes gcc
apt-get install --assume-yes g++
apt-get install --assume-yes make
make -C scikit-learn/ clean inplace

export PYTHONPATH=${PYTHONPATH}:~/scikit-learn

