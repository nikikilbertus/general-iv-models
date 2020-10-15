#!/usr/bin/env bash

python3 src/run.py --dataset=synthetic --equations=lin1 --output_dir=results/ --response_type=poly --dim_theta=2 --output_name=lin1-linear
python3 src/run.py --dataset=synthetic --equations=lin1 --output_dir=results/ --response_type=poly --dim_theta=3 --output_name=lin1-quadratic
python3 src/run.py --dataset=synthetic --equations=lin1 --output_dir=results/ --response_type=mlp --dim_theta=7 --output_name=lin1-mlp

python3 src/run.py --dataset=synthetic --equations=lin2 --output_dir=results/ --response_type=poly --dim_theta=2 --output_name=lin2-linear
python3 src/run.py --dataset=synthetic --equations=lin2 --output_dir=results/ --response_type=poly --dim_theta=3 --output_name=lin2-quadratic
python3 src/run.py --dataset=synthetic --equations=lin2 --output_dir=results/ --response_type=mlp --dim_theta=7 --output_name=lin2-mlp

