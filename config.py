from dotenv import load_dotenv
from datasets import load_dataset
import os 
load_dotenv()


mistral_api_key = os.getenv('MISTRAL_API_KEY')
api_key = os.getenv('API_KEY')
general_endpoint = 'https://api.ohmygpt.com/v1/'
models_close = ['claude-opus-4-1-20250805','claude-3-5-haiku-latest','gemini-1.5-flash-latest','gemini-1.5-pro-latest','TA/google/gemma-2-9b-it','TA/google/gemma-2-27b-it','gpt-3.5-turbo-0301','gpt-3.5-turbo-0613','gpt-3.5-turbo-0125','gpt-4-0125-preview','gpt-4o','gpt-4-turbo-2024-04-09','gpt-4o-2024-08-06','TA/meta-llama/Llama-2-7b-chat-hf','TA/meta-llama/Llama-2-13b-chat-hf','TA/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo','TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo','TA/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo','meta-llama/Llama-3.3-70B-Instruct-Turbo','gpt-5-chat-latest']
models_close_2 = ['TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo']
models_mistral = ['mistral-large-latest','mistral-small-latest','mistral-medium-latest']
#finished: gemini-2.0-flash llama3.3-70b claude-3.5-sonnet
#not avaliable: gemini-1.0-pro llama2-70b llama3-8b llama3-70b 
models_opensource = ['google/gemma-7b-it']
models_ollama = ['llama3:8b','llama3:70b']

categories = ['deception','harassment','harmful','hate','illegal','privacy','self-harm','sexual','unethical','violence','all']

dataset = load_dataset('bench-llm/or-bench','or-bench-hard-1k')['train']



#pipelines config
pipelines_categories = ['fraud','politics','race','religion','suicide','terrorism','violence','pornography_sexual_minors']

#ai/ml api
aiml_endpoint = "https://api.aimlapi.com/v1"
aiml_api_key = os.getenv('AIML_API_KEY')
