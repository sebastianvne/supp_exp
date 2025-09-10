class Pipeline:
    def __init__(self,model,judger_llm,judger_sampling_params):
        self.model = model
        self.judger_llm = judger_llm
        self.judger_sampling_params = judger_sampling_params
        #model client
    def process_category(self,category,dataset,output_folder):
        pass
    def _response(self,prompt):
        return self.model._response(prompt)