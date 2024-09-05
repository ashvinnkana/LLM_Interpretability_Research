import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GROQ:
    def __init__(self):
        self.model = None
        self.groq_client = Groq(api_key=os.getenv('GROQ_CONN_APIKEY'))

    def set_model(self, model):
        self.model = model

    def generate_response(self, query: str, message: str):
        messages = [
            {"role": "system", "content": message},
            {"role": "user", "content": query}
        ]
        # generate response
        chat_response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=messages,
            # focussed responses
            temperature=0, # controlled randomness
            top_p=0 # controlled diversity
        )

        return chat_response.choices[0].message.content
