from config import is_refuse_prompt
import re
class Pipeline:
    def __init__(self,model,judger_llm,judger_sampling_params,refusal_judger_llm):
        self.model = model
        self.judger_llm = judger_llm
        self.judger_sampling_params = judger_sampling_params
        self.refusal_judger_llm = refusal_judger_llm
        #model client

    def _is_refuse(self,prompt,response):
        prompt = is_refuse_prompt.format(prompt=prompt,response=response)
        is_refuse_response = self.refusal_judger_llm._response(prompt)
        items = re.findall(r'\[\[(.*?)\]\]', is_refuse_response)
        content = items[0] if items else None
        if content is None:
            print(f"[Warning]Not Found Content")
            return False
        if content == 'indirect_refusal' or content == 'direct_refusal':
            return True
        elif content == 'direct_answer':
            return False
        else:
            print(f"[Warning] Invalid output")
            return False
    
    def process_category(self,category,dataset,output_folder):
        pass
    def _response(self,prompt):
        return self.model._response(prompt)