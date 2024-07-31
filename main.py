import os
from scripts import extract_file, clean_text
from utils.llm_handler import LLM
from utils.pinecone_client import PINECONE
from utils.embedding_model import EMBEDDER
from utils.groq_client import GROQ
from utils import constants
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# access environmental variables
vectordb_connect_key = os.getenv('PINECONE_CONN_APIKEY')
fastai_interface_conn_apikey = os.getenv('GROQ_CONN_APIKEY')

# setup
embedder = EMBEDDER('dwzhu/e5-base-4k')

vectordb = PINECONE(vectordb_connect_key, 'llm-research', embedder.get_dimensions())
vectordb.connect()

fastai_interface = GROQ(fastai_interface_conn_apikey)
fastai_interface.set_model('llama3-70b-8192')


def get_response(query, top_k):
    query_embeds = embedder.encode(query)
    docs = vectordb.get_docs(query_embeds, top_k)

    return fastai_interface.generate_response(query, docs, 'AI')


def extract_pdf():
    # Extract PDF Contents
    data = extract_file.extract_pdf('./data/unstructured_data/wa_legislation_files/Limitation_Act_2005.pdf')
    data = clean_text.basic(data)

    # print content of the file
    print(data)


def generate_reference_text():
    # TODO: Ask a question passing this extracted text to LLM

    legal_llm = LLM(constants.aus_legal_llm)
    tokens = legal_llm.tokenize('Section 51 of the Constitution provides')
    print(legal_llm.generate_response(tokens))


def main():
    return


if __name__ == '__main__':
    main()
