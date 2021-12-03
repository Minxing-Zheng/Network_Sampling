#!/bin/bash

tar -xvzf model_save.tar.gz
rm model_save.tar.gz

# untar your Python installation. Make sure you are using the right version!
tar -xzf python38.tar.gz
# (optional) if you have a set of packages (created in Part 1), untar them also
tar -xzf packages.tar.gz

export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

python3 -m pip install --target=$PWD/packages python-louvain

# run your script
python3 run.py $1 $2 $3 $4




