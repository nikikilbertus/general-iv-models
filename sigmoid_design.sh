#!/usr/bin/env bash

python3 src/run.py --dataset=synthetic --equations=np --output_dir=results/ --response_type=poly --dim_theta=4 --slack_abs=0.1 --output_name=sigmoid-cubic
python3 src/run.py --dataset=synthetic --equations=np --output_dir=results/ --response_type=gp --dim_theta=7 --slack_abs=0.1 --output_name=sigmoid-gp
python3 src/run.py --dataset=synthetic --equations=np --output_dir=results/ --response_type=mlp --dim_theta=7 --slack_abs=0.1 --output_name=sigmoid-mlp

