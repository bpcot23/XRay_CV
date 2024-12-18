#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate XRay
conda env update --file requirements.yml --prune