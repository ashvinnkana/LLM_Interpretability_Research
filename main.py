import os
from scripts import extract_file
from scripts.preprocess_text import clean, chunk_by_word_limit, convert_to_dataset
from utils.pipeline_llm_handler import LLM
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

vectordb = PINECONE(vectordb_connect_key,
                    'llm-research',
                    embedder.get_dimensions(),
                    'aws', 'us-east-1')
vectordb.connect()

fastai_interface = GROQ(fastai_interface_conn_apikey)


def preprocess_data(file_path):
    # Extract PDF Contents
    data = extract_file.extract_pdf_to_text(file_path)
    return convert_to_dataset(chunk_by_word_limit(clean(data)))


def process_and_upsert_pdf_data(file_path):
    dataset = preprocess_data(file_path)
    embedder.encode_upsert_vectordb(dataset, 10, vectordb)


def get_response(query, docs, topic, llm):
    fastai_interface.set_model(constants.groq_supported_llm_list[llm]['model_id'])
    return fastai_interface.generate_response(query, docs, topic)


def non_legal_llm_responses(question, docs):
    # LLM : 0 = llama, 1 = mixtral, 2 = gemma
    for index, llm in enumerate(constants.groq_supported_llm_list):
        print(f'{llm['model_id']}\n{get_response(question, docs, 'LAW', index)}\n')


def legal_llm_response(question, docs):
    legal_llm = LLM(constants.aus_legal_llm, 'question-answering')

    print(f'{constants.aus_legal_llm}\n{legal_llm.generate(question, docs)}\n')


def main():
    process_and_upsert_pdf_data('./data/unstructured_data/wa_legislation_files/Limitation_Act_2005.pdf')

    question = "What is the section that states the limitation period for a continuous adverse possession in WA?"
    print(f"question: {question}\n")

    query_embeds = embedder.encode(question)
    docs = vectordb.get_docs(query_embeds, 5)

    non_legal_llm_responses(question, docs)
    legal_llm_response(question, docs)


if __name__ == '__main__':
    main()
