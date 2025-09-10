import pandas as pd
from datasets import Dataset
import os
import json

def construct_dataset_from_dir(dir_path):
    data = []
    for subdir in os.listdir(dir_path):
        for file in os.listdir(os.path.join(dir_path, subdir)):
            if file == 'generated_cases.json':
                with open(os.path.join(dir_path, subdir, file), "r") as f:
                    temp = json.load(f)
                    for item in temp:
                        item['Category'] = subdir
                        data.append(item)
    dataset = Dataset.from_list(data)
    return dataset

if __name__ == "__main__":
    #test and dump
    dataset = construct_dataset_from_dir("/home/23099359d/Enhancing_LLAMA_Against_Language_Jailbreaks/data/benchmarks/SAP/SAP200")
    print(dataset)
