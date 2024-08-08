import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from utils.constants import pinecone_cloud, pinecone_region, pinecone_dimension

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

    def get_docs(self, query, top_k) -> list[str]:
        res = self.index.query(vector=query, top_k=top_k, include_metadata=True)

        # get doc text
        try:
            docs = [f'CHUNK SOURCED FROM {chunk['metadata']['source']}\n\n{chunk['metadata']['content']}'
                    for chunk in res["matches"]]
        except KeyError:
            docs = [chunk['metadata']['content'] for chunk in res["matches"]]
        return docs
