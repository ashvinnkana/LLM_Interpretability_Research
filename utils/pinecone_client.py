import time
from pinecone import Pinecone, ServerlessSpec


class PINECONE:
    def __init__(self, apikey, index_name, dimensions):
        self.index = None
        self.index_name = index_name
        self.dims = dimensions
        self.db_client = Pinecone(api_key=apikey)
        self.spec = ServerlessSpec(
            cloud="aws", region="us-east-1"
        )

        existing_indexes = [
            index_info["name"] for index_info in self.db_client.list_indexes()
        ]

        # check if index already exists (it shouldn't if this is first time)
        if index_name not in existing_indexes:
            # if it does not exist, create index
            self.db_client.create_index(
                index_name,
                dimension=dimensions,
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
        docs = [x['metadata']['content'] for x in res["matches"]]
        return docs


