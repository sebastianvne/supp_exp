from openai import OpenAI
from config import general_endpoint,api_key,is_refuse_prompt
import re

class CloseSourceLLM:
    def __init__(self,model_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key,base_url=general_endpoint)

    def _is_refuse(self,prompt,response):
        prompt = is_refuse_prompt.format(prompt=prompt,response=response)
        is_refuse_response = self._response(prompt)
        match = re.findall(r'\[\[.*?\]\]', is_refuse_response)
        result = match.group(0) if match else None
        if 'refusal' in result:
            return True
        else:
            return False

    def _response(self,prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {str(e)}"
    def process_category(self,category,dataset):
        pass