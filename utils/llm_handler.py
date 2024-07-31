from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

set_seed(42)


class LLM:
    def  __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize(self, input_text):
        return self.tokenizer(input_text,
                              return_tensors='pt',
                              truncation=True,
                              max_length=1024)

    def generate_response(self, tokens):
        response = self.model.generate(tokens,
                                       max_length=150,
                                       num_beams=5,
                                       early_stopping=True)

        return self.tokenizer_decode(response)

    def tokenizer_decode(self, response):
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
