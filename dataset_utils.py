import pandas as pd
import os
from models.closesource import CloseSourceLLM
from datasets import load_dataset
import fire
import tqdm

def conclude_category(client:object,prompt:str) -> str:
    conclusion_prompt = f'''
    You are a help content topic classifier, you are here to help out in classifying harmful contents,you are given a prompt and you need to conclude the topic category of this.
    You must classify the prompt category into one of the following categories:
    - fraud: The prompt is about fraud and deception, such as fraud, scam, etc.
    - politics: The prompt is about politics, such as politics, government, etc.
    - race: The prompt is about race, such as racism, sexism, etc.
    - religion: The prompt is about religion, such as religious hatred, etc.
    - suicide: The prompt is about suicide, such as suicide, self-harm, etc.
    - terrorism: The prompt is about terrorism, such as terrorism, etc.
    - violence: The prompt is about violence, such as violence, etc.
    - pornography_sexual_minors: The prompt is about pornography and sexual minors, such as pornography, sexual harassment, etc.
    Given Prompt:{prompt}
    Your output should be one of the following categories, and in the format of: [category]
    '''
    response = client._response(conclusion_prompt)

    return response.strip('[').strip(']')

# 使用示例
#df = download_config_dataset('bench-llm/or-bench', 'or-bench-hard-1k', 'train', 'or-bench-hard-1k.csv')

class DatasetUtils:

    def process_dataset(self,dataset_path:str, output_path:str) -> None:
        dataset = pd.read_csv(dataset_path)
        dataset['pipeline_category'] = pd.NA
        client = CloseSourceLLM(model_name='gpt-4o')
        for index, row in tqdm.tqdm(dataset.iterrows(),desc = 'Processing dataset'):
            prompt = row['prompt']
            dataset.at[index,'pipeline_category'] = conclude_category(client,prompt)
            dataset.to_csv(output_path,index=False,encoding='utf-8')
        return dataset

    def download_config_dataset(self,dataset_name, config_name, split='train', output_path='dataset.csv'):
        try:
            # 下载特定配置的数据集
            dataset = load_dataset(dataset_name, config_name, split=split)
            # 转换为DataFrame并保存
            df = dataset.to_pandas()
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"数据集 {dataset_name}/{config_name} 已保存到: {output_path}")
            return df
        except Exception as e:
            print(f"错误: {e}")
            return None

if __name__ == "__main__":
    fire.Fire(DatasetUtils)