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

    def generate_response(self, query: str, docs: str, topic: str):
        system_message = (
            f"You are a helpful assistant that answers questions in two sentences about {topic} using the "
            f"context provided below.\n\nCONTEXT:\n{docs}"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        # generate response
        chat_response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return chat_response.choices[0].message.content
