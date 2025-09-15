from pipelines.pipeline import Pipeline
from pipelines.explicit_pipeline import ExplicitPipeline
from pipelines.implicit_pipeline import ImplicitPipeline
import pandas
import os
import csv
from tqdm import tqdm

class ComposedPipeline(Pipeline):
    def __init__(self,model,judger_llm,judger_sampling_params,refusal_judger_llm,is_rewrite = False,max_rewrite_epoch = 3):
        super().__init__(model,judger_llm,judger_sampling_params,refusal_judger_llm)
        self.explicit_pipeline = ExplicitPipeline(model,judger_llm,judger_sampling_params,refusal_judger_llm)
        self.implicit_pipeline = ImplicitPipeline(model,judger_llm,judger_sampling_params,refusal_judger_llm,is_rewrite,max_rewrite_epoch)
    def process_single_both(self,item,category,output_folder):
        prompt = item.iloc[0]
        explicit_output = self.explicit_pipeline.process_single(item,category,output_folder,no_response = False,is_save = True)
        implicit_output = self.implicit_pipeline.process_single(item,category,output_folder,implicit_response = None,no_response = False,is_save = True)
        if implicit_output is None or explicit_output is None:
            return None
        
        if ( implicit_output['is_filtered_by_basemodel'] != [1]) and ( implicit_output['is_filtered_by_pipeline'] != [1]) and (explicit_output['is_filtered_by_pipeline'] != [1]) and (explicit_output['is_filtered_by_basemodel'] != [1]):
            composed_output = {
                "prompt": [prompt],
                "explicit_output": [explicit_output],
                "implicit_output": [implicit_output],
                "final_response": [implicit_output['final_response']],
                "is_filtered_by_composed_pipeline": [0],
                "is_filtered_by_implicit_pipeline": [0],
                "is_filtered_by_implicit_basemodel": [0],
                "is_filtered_by_explicit_pipeline": [0],
                "is_filtered_by_explicit_basemodel": [0],
                "category": [category],
            }
            return composed_output
        else:
            composed_output = {
                "prompt": [prompt],
                "explicit_output": [explicit_output],
                "implicit_output": [implicit_output],
                "final_response": [implicit_output['final_response']],
                "is_filtered_by_composed_pipeline": [1],
                "is_filtered_by_implicit_pipeline": implicit_output['is_filtered_by_pipeline'],
                "is_filtered_by_implicit_basemodel": implicit_output['is_filtered_by_basemodel'],
                "is_filtered_by_explicit_pipeline": explicit_output['is_filtered_by_pipeline'],
                "is_filtered_by_explicit_basemodel": explicit_output['is_filtered_by_basemodel'],
                "category": [category],
            }
            return composed_output

    def process_category(self,category,dataset,output_folder):
        category_err_flag = False
        try:
            filtered_dataset = dataset[dataset['category'] == category]
        except:
            print(f"category {category} not found in dataset")
            category_err_flag = True
            filtered_dataset = dataset
        print(f'running {category}...')
        outputs = []
        for _,item in tqdm(filtered_dataset.iterrows(),desc = 'running prompts'):
            output_data = self.process_single_both(item,category,output_folder)
            if output_data is None:
                continue
            df = pandas.DataFrame(output_data)
            to_path = self.model.model_name.replace("/","-")
            df.to_csv(f'{output_folder}/composed_pipeline/{to_path}_{category}_{self.implicit_pipeline.is_rewrite}_{self.implicit_pipeline.max_rewrite_epoch}_results.csv',mode='a',header=not os.path.isfile(f'{output_folder}/composed_pipeline/{to_path}_{category}_{self.implicit_pipeline.is_rewrite}_{self.implicit_pipeline.max_rewrite_epoch}_results.csv'),index=False)
            outputs.append(output_data)
        return outputs,category_err_flag

'''
    def process_single(self,item,category,output_folder):
        explicit_output = self.explicit_pipeline.process_single(item,category,output_folder,no_response = False,is_save = False)
        if explicit_output is None:
            print("[Warning] Explicit Output is None")
            return None
        else:
            if explicit_output['is_filtered_by_pipeline'] == 0:
                implicit_output = self.implicit_pipeline.process_single(item,category,output_folder,implicit_response = explicit_output['response'],no_response = True,is_save = False)
                if implicit_output is None:
                    print("[Warning] Implicit Output is None")
                    return None
            else:
                implicit_output = None
                composed_output = {
                    "prompt": [prompt],
                    "explicit_output": [explicit_output],
                    "implicit_output": [implicit_output],
                    "final_response": [explicit_output['response']],
                    "is_filtered_by_composed_pipeline": [1],
                    "is_filtered_by_implicit_pipeline": [0],
                    "is_filtered_by_implicit_basemodel": [0],
                    "is_filtered_by_explicit_pipeline": [1],
                    "is_filtered_by_explicit_basemodel": [1],
                    "category": [item['category']],
                }
                return composed_output
        
        
        
        if (implicit_output['is_filtered_by_basemodel'] != [1]) and (implicit_output['is_filtered_by_pipeline'] != [1]):
            composed_output = {
                "prompt": [item['prompt']],
                "explicit_output": [explicit_output],
                "implicit_output": [implicit_output],
                "final_response": [implicit_output['final_response']],
                "is_filtered_by_composed_pipeline": [0],
                "is_filtered_by_implicit_pipeline": [0],
                "is_filtered_by_implicit_basemodel": [0],
                "is_filtered_by_explicit_pipeline": [0],
                "is_filtered_by_explicit_basemodel": [0],
                "category": [category],
            }
            return composed_output
        else:
            composed_output = {
                "prompt": [prompt],
                "explicit_output": [explicit_output],
                "implicit_output": [implicit_output],
                "final_response": [implicit_output['final_response']],
                "is_filtered_by_composed_pipeline": [1],
                "is_filtered_by_implicit_pipeline": [1],
                "is_filtered_by_implicit_basemodel": [1],
                "is_filtered_by_explicit_pipeline": [1],
                "is_filtered_by_explicit_basemodel": [1],
                "category": [category],
            }
            return composed_output

'''
