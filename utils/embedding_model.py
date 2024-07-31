from semantic_router.encoders import HuggingFaceEncoder
from tqdm.auto import tqdm


class EMBEDDER:
    def __init__(self, name):
        self.name = name
        self.encoder = HuggingFaceEncoder(name=name)

    def get_encoder_device(self):
        return self.encoder.device

    def get_dimensions(self):
        embeds = self.encoder(["TEST TO GET DIMENSIONS"])
        return len(embeds[0])

    def encode(self, text):
        return self.encoder([text])

    def encode_upsert_vectordb(self, data, batch_size, db_client):
        for i in tqdm(range(0, len(data), batch_size)):

            # find end of batch
            i_end = min(len(data), i + batch_size)
            # create batch
            batch = data[i:i_end]

            # create embeddings
            chunks = [f'{x["content"]}' for x in batch["metadata"]]
            embeds = self.encoder(chunks)

            assert len(embeds) == (i_end - i)

            vectors = list(zip(batch["id"], embeds, batch["metadata"]))

            db_client.upsert(vectors)
