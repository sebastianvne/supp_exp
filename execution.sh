#!/bin/bash

nohup python3 pipelines_benchmarking.py \
--pipeline_type composed \
--model gpt-5-chat-latest \
--category all \
--is_dataset_local False \
--output_folder ./results/pipelines/HarmBench \
--input_path bench-llms/or-bench \
--subset or-bench-hard-1k \
--split train > ./log/HarmBench_composed_pipeline_gpt-5-chat-latest.log 2>&1 &
