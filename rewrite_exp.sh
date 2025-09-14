#!/bin/bash

echo "input log file: (please following the format: dataset_used + pipeline_type + model_used)"
read log_file

nohup python3 pipelines_benchmarking.py \
--pipeline_type composed \
--model gpt-3.5-turbo-0125 \
--category all \
--is_dataset_local True \
--output_folder ./results/pipelines/or-bench-hard-1k \
--input_path ./datasets/or-bench/or-bench-hard-1k.csv  \
--is_rewrite True \
--max_rewrite_epoch 3 \ > ./log/${log_file} 2>&1 &

echo "running..."
echo "pid: [$!]"