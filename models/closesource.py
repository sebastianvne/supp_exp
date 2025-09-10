from openai import OpenAI
from config import general_endpoint,api_key,aiml_api_key,aiml_endpoint
class CloseSourceLLM:
    def __init__(self,model_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key,base_url=general_endpoint)

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