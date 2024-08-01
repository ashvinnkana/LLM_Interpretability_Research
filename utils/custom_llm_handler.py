from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

set_seed(42)


class LLM:
    def  __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Check if GPU is available and move the model to GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def tokenize(self, input_text):
        return self.tokenizer(input_text,
                              return_tensors='pt',
                              truncation=True,
                              max_length=512)

    def generate_response(self, tokens):
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        response = self.model.generate(tokens["input_ids"],
                                       attention_mask=tokens["attention_mask"],
                                       pad_token_id=self.tokenizer.eos_token_id,
                                       max_new_tokens=150,
                                       num_beams=5,
                                       early_stopping=True)

        return self.tokenizer_decode(response)

    def tokenizer_decode(self, response):
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
