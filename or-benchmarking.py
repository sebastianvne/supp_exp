from config import models_mistral,dataset,categories
import fire
from models.mistral import MistralBenchmarking
import json
import tqdm
from models.finetuned_llama import FinetunedLlama

def execution(model,client,cats):
    benchmarking_client = client
    summary = []
    total_rejection_count = 0
    total_total_count = 0
    for cat in tqdm.tqdm(cats,desc = 'running prompt categories'):
        cat_sum = benchmarking_client.process_category(cat,dataset)
        summary.append(cat_sum)
        total_rejection_count += cat_sum['rejection_count']
        total_total_count += cat_sum['total_count']
    print(f'model: {model}' + '-'*50 + '\n')
    print(f'total_rejection_count: {total_rejection_count}')
    print(f'total_total_count: {total_total_count}')
    print(f'total_rejection_rate: {total_rejection_count/total_total_count}')
    print('\n')
    with open(f'./models/{model}_summary.json','w') as f:
        json.dump(summary,f,indent=4)

def main(model,category):
    if category == 'all':
        cats = categories[:-1]
    else:
        cats = [category]

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
    for model,client in tqdm.tqdm(zip(models_toexec,clients),desc = 'running models'):
        print(f'running {model}...')
        execution(model,client,cats)
        
if __name__ == "__main__":
    fire.Fire(main)