#!/bin/bash

nohup python3 pipelines_benchmarking.py \
--pipeline_type composed \
--model TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
--category all \
--is_dataset_local True \
--output_folder ./results/pipelines/SAP \
--input_path ./datasets/SAP/dataset.csv \
--subset None \
--split None > ./log/SAP_composed_pipeline_Meta-Llama-3.1-70B-Instruct-Turbo.log 2>&1 &
