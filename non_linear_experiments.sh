#!/usr/bin/env bash

python3 src/run.py --dataset=synthetic --equations=quad1 --response_type=poly --dim_theta=2 --output_name=quad1-linear
python3 src/run.py --dataset=synthetic --equations=quad1 --response_type=poly --dim_theta=3 --output_name=quad1-quadratic
python3 src/run.py --dataset=synthetic --equations=quad1 --response_type=mlp --dim_theta=7 --output_name=quad1-mlp

python3 src/run.py --dataset=synthetic --equations=quad2 --response_type=poly --dim_theta=2 --output_name=quad2-linear
python3 src/run.py --dataset=synthetic --equations=quad2 --response_type=poly --dim_theta=3 --output_name=quad2-quadratic
python3 src/run.py --dataset=synthetic --equations=quad2 --response_type=mlp --dim_theta=7 --output_name=quad2-mlp

