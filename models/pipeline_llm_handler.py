from transformers import pipeline, set_seed
import torch

set_seed(42)


class LLM:
    def __init__(self):
        self.model = None
        self.task = "question-answering"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use GPU if available
        else:
            self.device = torch.device("cpu")  # Use CPU otherwise

    def set_model(self, model):
        self.model = model

    def generate_response(self, query: str, message: str):
        generator = pipeline(self.task,
                             device=self.device,
                             model=self.model,
                             tokenizer=self.model)

        return generator(query, message, max_length=500)['answer']
