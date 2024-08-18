import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIGPT:
    def __init__(self):
        self.model = None
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def set_model(self, model):
        self.model = model

    def generate_response(self, query: str, message: str):
        messages = [
            {"role": "system", "content": message},
            {"role": "user", "content": query}
        ]
        # generate response
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return chat_response.choices[0].message['content']
