import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from utils.constants import pinecone_cloud, pinecone_region, pinecone_dimension
from utils.functions import clean_for_embeds

load_dotenv()


class PINECONE:
    def __init__(self, index_name):
        self.index = None
        self.index_name = index_name
        self.dims = pinecone_dimension
        self.db_client = Pinecone(api_key=os.getenv('PINECONE_CONN_APIKEY'))
        self.spec = ServerlessSpec(
            cloud=pinecone_cloud, region=pinecone_region
        )

        existing_indexes = [
            index_info["name"] for index_info in self.db_client.list_indexes()
        ]

        # check if index already exists (it shouldn't if this is first time)
        if index_name not in existing_indexes:
            # if it does not exist, create index
            self.db_client.create_index(
                index_name,
                dimension=pinecone_dimension,
                metric='cosine',
                spec=self.spec
            )
            # wait for index to be initialized
            while not self.db_client.describe_index(index_name).status['ready']:
                time.sleep(1)

    def connect(self):
        # connect to index
        self.index = self.db_client.Index(self.index_name)
        time.sleep(1)

    def index_status(self):
        return self.index.describe_index_stats()

    def upsert(self, vectors):
        self.index.upsert(vectors=vectors)

    def get_docs(self, query_str, query_vector, top_k) -> list[str]:
        res = self.index.query(vector=query_vector, top_k=top_k*2, include_metadata=True)

        tokenized_query = clean_for_embeds(query_str).split(" ")
        if 'title' in res['matches'][0]['metadata']:
            # v2
            docs = [{'title': chunk['metadata']['title'], 'content': chunk['metadata']['content']}
                    for chunk in res["matches"]]
            tokenized_corpus = [clean_for_embeds(f"{doc['title']} : {doc['content']}").split(" ") for doc in docs]

        elif 'spans' in res['matches'][0]['metadata']:
            # v2.1
            docs = [{'spans': chunk['metadata']['spans'],
                     'content': chunk['metadata']['content'],
                     'id': chunk['id']}
                    for chunk in res["matches"]]
            tokenized_corpus = [clean_for_embeds(doc['content']).split(" ") for doc in docs]

        elif 'text' in res['matches'][0]['metadata']:
            # unstructured io
            docs = [{'text': chunk['metadata']['text'],
                     'Filename': chunk['metadata']['Filename'],
                     'Filetype': chunk['metadata']['Filetype'],
                     'Page Number': chunk['metadata']['Page Number'],}
                    for chunk in res["matches"]]
            tokenized_corpus = [clean_for_embeds(f"{doc['Filename']} {doc['Page Number']} {doc['text']}").split(" ") for doc in docs]

        else:
            # other versions
            docs = [chunk['metadata']['content'] for chunk in res["matches"]]
            tokenized_corpus = [clean_for_embeds(doc).split(" ") for doc in docs]

        bm25 = BM25Okapi(tokenized_corpus)
        top_n = bm25.get_top_n(tokenized_query, docs, n=top_k)
        return top_n
