#!/usr/bin/env bash
set -e
cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/ephys
source ~/Documents/PYTHON/envs/iblenv/bin/activate
python ephys.py ks2_qc $1
