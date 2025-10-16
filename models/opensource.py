from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import tqdm
import os
from config import is_refuse_prompt

class OpenSourceLLM:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        
        self.tokenizer = None
        self.model = None

        # 导入模型和分词器
        #self.lazy_init()

    def lazy_init(self):
        # 导入模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _response(self, prompt, max_new_tokens=32, temperature=0.1):
        """生成响应"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if temperature == 0.0:
                # 使用贪婪解码，确定性生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 贪婪解码
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                # 使用采样生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # 移除输入部分，只返回生成的内容
        response = response[len(prompt):].strip()
        
        return response

    def _is_refuse(self, prompt, response):
        """判断是否拒绝回答"""
        refuse_prompt = is_refuse_prompt.format(prompt=prompt, response=response)
        refuse_response = self._response(refuse_prompt, max_new_tokens=10)
        # 这里需要根据您的具体逻辑来判断是否拒绝
        match = re.findall(r'\[\[.*?\]\]', refuse_response)
        result = match.group(0) if match else None
        if 'refusal' in result:
            return True
        else:
            return False