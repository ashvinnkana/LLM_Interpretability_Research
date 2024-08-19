import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ANTHROPIC:
    def __init__(self):
        self.model = None
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def set_model(self, model):
        self.model = model

    def generate_response(self, query: str, message: str):
        messages = [
            {"role": "user", "content": message + query}
        ]
        # generate response
        chat_response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=1024
        )

        return chat_response.content[0].text
