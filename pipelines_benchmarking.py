from pipelines.explicit_pipeline import ExplicitPipeline
from pipelines.implicit_pipeline import ImplicitPipeline
from pipelines.pipeline import Pipeline
from pipelines.tools import construct_dataset_from_dir
from config import pipelines_categories,models_mistral,categories,models_close
from models.finetuned_llama import FinetunedLlama
from models.mistral import MistralLLM
import fire
import tqdm
from datasets import load_dataset
import pandas
from pipelines.config import PROMPT_TEMPLATE
from vllm import LLM,SamplingParams
import os
import time
from models.closesource import CloseSourceLLM
from pipelines.composed_pipeline import ComposedPipeline
def init_judger():
    CLS_MODEL = "cais/HarmBench-Llama-2-13b-cls"
    llm = LLM(model=CLS_MODEL, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
    )
    return llm, sampling_params

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

def main(pipeline_type,model,category,is_dataset_local,output_folder,input_path,subset = None,split = None):
    
    #dataset = pandas.read_csv('./datasets/SAP/dataset.csv')
    dataset = get_dataset(input_path,is_local=is_dataset_local,subset=subset,split=split)
    if category == 'all':
        categories_toexec = [item for item in pipelines_categories if item != 'violence' and item != 'fraud' and item != 'politics' and item != 'race' ]
    else:
        categories_toexec = [category]

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
    if model in models_close:
        models_toexec = [model]
        clients = [CloseSourceLLM(model)]

    llm, sampling_params = init_judger()
    for model,client in tqdm.tqdm(zip(models_toexec,clients),desc = 'running models'):
        print(f'running {model}...')
        pipeline = Pipeline(client,llm,sampling_params)
        if pipeline_type == 'explicit':
            pipeline = ExplicitPipeline(client,llm,sampling_params)
        elif pipeline_type == 'implicit':
            pipeline = ImplicitPipeline(client,llm,sampling_params)
        elif pipeline_type == 'composed':
            pipeline = ComposedPipeline(client,llm,sampling_params)
        for cat in categories_toexec:
            pipeline.process_category(cat,dataset,output_folder)
if __name__ == "__main__":
    fire.Fire(main)