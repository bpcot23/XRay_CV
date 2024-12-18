#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda remove -n XRay --all
conda env create -f requirements.yml
conda activate XRay
