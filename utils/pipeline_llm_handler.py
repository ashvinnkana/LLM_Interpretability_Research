from transformers import pipeline, set_seed
import torch

set_seed(42)


class LLM:
    def __init__(self, model, task):
        self.model = model
        self.task = task

        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use GPU if available
        else:
            self.device = torch.device("cpu")  # Use CPU otherwise

        self.generator = pipeline(task,
                                  torch_dtype=torch.float32,
                                  device=self.device,
                                  model=model)

    def generate(self, question, docs):
        return self.generator(question, "\n---\n".join(docs), max_length=500)['answer']
