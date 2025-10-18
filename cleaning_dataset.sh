#!/bin/bash
nohup python3 ./dataset_utils.py \
process_dataset \
--dataset_path ./benchmark_datasets/or-bench/or-bench-toxic.csv \
--output_path ./benchmark_datasets/or-bench/or-bench-toxic-processed.csv > ./log/cleaning_dataset.log 2>&1 &

echo "[$!] Cleaning dataset pid: $!"