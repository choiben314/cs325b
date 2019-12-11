#!/bin/sh

cd code

version=$(which python)
echo "Using python binary at $version"

python submit.py

echo "See provided text files for installation/setup/run instructions."

cd ..
