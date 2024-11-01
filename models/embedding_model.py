import torch
from semantic_router.encoders import HuggingFaceEncoder
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils.functions import clean_for_embeds


class EMBEDDER:
    def __init__(self, name):
        self.name = name
        self.encoder = HuggingFaceEncoder(name=name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.device = device

        # Load the tokenizer separately using the model name
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def get_encoder_device(self):
        return self.encoder.device

    def get_dimensions(self):
        embeds = self.encoder(["TEST TO GET DIMENSIONS"])
        return len(embeds[0])

    def encode(self, text):
        return self.encoder([text])

    def encode_upsert_vectordb(self, data, batch_size, db_client, version):
        for i in tqdm(range(0, len(data), batch_size)):

            # find end of batch
            i_end = min(len(data), i + batch_size)
            # create batch
            batch = data[i:i_end]

            # create embeddings
            if version == 'v2':
                chunks = [clean_for_embeds(f'{bat["title"].split(' >> ')[-1]} {bat["content"]}')
                          for bat in batch["metadata"]]
            elif version == 'unstruct-io':
                chunks = [clean_for_embeds(f'{bat["Filename"]} {bat["Page Number"]} {bat["text"]}')
                          for bat in batch["metadata"]]
            else:
                chunks = [clean_for_embeds(f'{bat["content"]}')
                          for bat in batch["metadata"]]

            embeds = self.encoder(chunks)

            assert len(embeds) == (i_end - i)

            vectors = list(zip(batch["id"], embeds, batch["metadata"]))

            db_client.upsert(vectors)

    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)
