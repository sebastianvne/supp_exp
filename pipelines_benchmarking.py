from pipelines.explicit_pipeline import ExplicitPipeline
from pipelines.implicit_pipeline import ImplicitPipeline
from pipelines.pipeline import Pipeline
from pipelines.tools import construct_dataset_from_dir
from config import or_bench_categories,models_mistral,models_close,models_open
from models.finetuned_llama import FinetunedLlama
from models.mistral import MistralLLM
import fire
import tqdm
from datasets import load_dataset
import pandas
from pipelines.config import PROMPT_TEMPLATE
#from vllm import LLM,SamplingParams 
#为了保证在非vllm环境下的运行，使用transformer改写这些方法
import os
import time
from models.closesource import CloseSourceLLM
from pipelines.composed_pipeline import ComposedPipeline
from models.opensource import OpenSourceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
'''
def init_judger():
    CLS_MODEL = "cais/HarmBench-Llama-2-13b-cls"
    llm = LLM(model=CLS_MODEL, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
    )
    return llm, sampling_params
'''
def init_judger():
    CLS_MODEL = "cais/HarmBench-Llama-2-13b-cls"
    tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL,use_fast = False)
    model = AutoModelForCausalLM.from_pretrained(CLS_MODEL,torch_dtype=torch.float16,device_map="auto")
    return model, tokenizer


def get_dataset(path,is_local,subset = None, split = None):

    '''
    TODO: add auto category detection
    '''
    if is_local:
        dataset = pandas.read_csv(path)
    else:
        #dataset = datasets.load_dataset('bench-llm/or-bench','or-bench-hard-1k')['train']
        dataset = load_dataset(path,subset)[split]
        dataset = dataset.to_pandas()
    #category detection:
    
    return dataset

def main(pipeline_type,model,category,is_dataset_local,output_folder,input_path,subset = None,split = None,is_rewrite = False,max_rewrite_epoch = 3):
    
    #dataset = pandas.read_csv('./datasets/SAP/dataset.csv')
    dataset = get_dataset(input_path,is_local=is_dataset_local,subset=subset,split=split)
    if category == 'all':
        categories_toexec = [item for item in or_bench_categories]
    else:
        categories_toexec = [category]
    clients = []
    models_toexec = []
    if model == 'finetuned_llama':
        models_toexec = [model]
        clients = [FinetunedLlama(model,original_model=False) for model in models_toexec]
    if model == 'original_llama':
        models_toexec = [model]
        clients = [FinetunedLlama(model,original_model=True) for model in models_toexec]
    if 'mistral' in model:
        if model == 'mistral-all':
            models_toexec = models_mistral
            clients = [MistralBenchmarking(model) for model in models_toexec]
        else:
            models_toexec = [model]
            clients = [MistralBenchmarking(model)]
    if model == 'close-all':
        models_toexec = models_close
        for model in models_toexec:
            try:
                clients.append(CloseSourceLLM(model))
            except Exception as e:
                print(f"Error: {e}")
                continue
    if model in models_close:
        models_toexec = [model]
        clients = [CloseSourceLLM(model)]

    if model == 'open-all':
        models_toexec = models_open
        for model in models_toexec:
            try:
                clients.append(OpenSourceLLM(model))
            except Exception as e:
                print(f"Error: {e}")
                continue
    elif model in models_open:
        models_toexec = [model]
        clients = [OpenSourceLLM(model)]

    if not models_toexec:
        raise ValueError(f"Unsupported model '{model}'. 请检查 config.py 中的模型列表或脚本入参。")


    llm, sampling_params = init_judger()
    refusal_judger_llm = CloseSourceLLM('gpt-4o')
    for model,client in tqdm.tqdm(zip(models_toexec,clients),desc = 'running models'):
        category_err_flag = False
        print(f'running {model}...')
        if model in models_open:
            client.lazy_init()
        pipeline = Pipeline(client,llm,sampling_params,refusal_judger_llm)
        if pipeline_type == 'explicit':
            pipeline = ExplicitPipeline(client,llm,sampling_params,refusal_judger_llm)
        elif pipeline_type == 'implicit':
            pipeline = ImplicitPipeline(client,llm,sampling_params,refusal_judger_llm,is_rewrite,max_rewrite_epoch)
        elif pipeline_type == 'composed':
            pipeline = ComposedPipeline(client,llm,sampling_params,refusal_judger_llm,is_rewrite,max_rewrite_epoch)
        for cat in categories_toexec:
            _,category_err_flag = pipeline.process_category(cat,dataset,output_folder)
        if category_err_flag:
            print(f"category error, have run the whole dataset, break the loop")
            break
if __name__ == "__main__":
    fire.Fire(main)