from groq import Groq


class GROQ:
    def __init__(self, apikey):
        self.model = None
        self.groq_client = Groq(api_key=apikey)

    def set_model(self, model):
        self.model = model

    def generate_response(self, query: str, docs: list[str], topic: str):
        system_message = (
            f"You are a helpful assistant that answers questions in one sentence about {topic} using the "
            "context provided below.\n\n"
            "CONTEXT:\n"
            "\n---\n".join(docs)
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
