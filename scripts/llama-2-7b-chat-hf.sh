#!/bin/bash
cd ../
export HF_HOME=/disk1/users/wangjh/hf_home
echo "input log file: (please following the format: dataset_used + pipeline_type + model_used)"
read -r log_file
default_log_file="or-bench-hard-1k-composed-llama2-7b-chat-hf.log"
log_file_trimmed="$(printf '%s' "$log_file" | tr -d '\r\n\t ')"
if [ -z "$log_file_trimmed" ]; then
  echo "no log file provided, fallback to default: ${default_log_file}"
  log_file="$default_log_file"
fi
export CUDA_VISIBLE_DEVICES=2
export HF_TOKEN=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
export HF_ENDPOINT=https://hf-mirror.com
nohup python3 pipelines_benchmarking.py \
--pipeline_type composed \
--model meta-llama/Llama-2-7b-chat-hf \
--category all \
--is_dataset_local True \
--output_folder ./results/pipelines/or-bench-hard-1k \
--input_path ./benchmark_datasets/or-bench/or-bench-hard-1k-processed.csv  \
--is_rewrite True \
--max_rewrite_epoch 3 > ./log/${log_file} 2>&1 &

echo "running..."
echo "pid: [$!]"
#or-bench-hard-1k-composed-llama2-7b-chat-hf.log
