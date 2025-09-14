from datasets import load_dataset
import pandas as pd

def download_config_dataset(dataset_name, config_name, split='train', output_path='dataset.csv'):
    """
    下载特定配置的数据集
    """
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

# 使用示例
df = download_config_dataset('bench-llm/or-bench', 'or-bench-hard-1k', 'train', 'or-bench-hard-1k.csv')