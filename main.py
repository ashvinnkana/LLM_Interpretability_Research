import logging.config

import pandas as pd
from groq import RateLimitError

from utils import strings
from utils import constants
from utils import logging_messages
from scripts import extract_data, structure_data

from models.pipeline_llm_handler import LLM
from models.pinecone_client import PINECONE
from models.embedding_model import EMBEDDER
from models.groq_client import GROQ
from utils.functions import download_nltk_resources
from utils.functions import save_preprocessed_data
from utils.functions import get_rouge_scores

# setup
download_nltk_resources()
logging.config.fileConfig(strings.logging_config_file)

embedder = EMBEDDER(constants.embedder)
fastai_interface = GROQ()

unstructured_vectordb = PINECONE(strings.unstructured_0_index)
json_structured_vectordb = PINECONE(strings.structured_JSON_1_index)


def legal_llm_response(question, docs):
    legal_llm = LLM(constants.aus_legal_llm, 'question-answering')
    print(f'{constants.aus_legal_llm}\n{legal_llm.generate(question, docs)}\n')


def create_results_dataframe(results, question_id, extract_version):
    logging.info(logging_messages.saving_responses)
    # Extract model names
    model_names = [model['model_id'] for model in constants.groq_supported_llm_list]

    # Define the multi-index columns
    metrics = ['rouge1', 'rougeL']
    columns = pd.MultiIndex.from_product([model_names, metrics], names=['model', 'metric'])

    # Prepare data dynamically
    data = []
    for key in results.keys():
        row = []
        for idx in range(len(constants.groq_supported_llm_list)):
            try:
                row.append(results[key][idx]['rouge1'])
                row.append(results[key][idx]['rougeL'])
            except TypeError as e:
                logging.error(logging_messages.error_saving_response.format(e))
                row.append(None)
                row.append(None)
                continue
        data.append(row)

    # Index for the rows
    index = list(results.keys())

    # Create the DataFrame
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_csv(strings.csv_results_path.format(question_id, extract_version))


def get_response(query, docs, topic, llm, ref_answer):
    logging.info(logging_messages.generating_response_with.format(constants.groq_supported_llm_list[llm]['model_id']))
    fastai_interface.set_model(constants.groq_supported_llm_list[llm]['model_id'])

    try:
        response = fastai_interface.generate_response(query, docs, topic)
        logging.info(logging_messages.response_display.format(response))
        return get_rouge_scores(ref_answer, response)
    except RateLimitError as e:
        logging.error(logging_messages.error_generating_response.format(e))
        return None


def non_legal_llm_responses(question, docs, ref_anw):
    response_scores = {}
    for index, llm in enumerate(constants.groq_supported_llm_list):
        response_scores[index] = get_response(question, docs, 'LAW', index, ref_anw)

    return response_scores


def generate_responses(question, ref_answer):
    response_scores = {}
    query_embeds = embedder.encode(question)

    # get responses for unstructured data
    logging.info(logging_messages.generating_response.format('UNSTRUCTURED'))
    unstructured_vectordb.connect()
    unstructured_docs = unstructured_vectordb.get_docs(query_embeds, 5)
    response_scores['unstructured'] = non_legal_llm_responses(question, unstructured_docs, ref_answer)

    # get responses for json_structured data
    logging.info(logging_messages.generating_response.format('JSON-STRUCTURED'))
    json_structured_vectordb.connect()
    json_structured_docs = json_structured_vectordb.get_docs(query_embeds, 5)
    response_scores['json-structured'] = non_legal_llm_responses(question, json_structured_docs, ref_answer)

    logging.info(logging_messages.main_divider)
    return response_scores


def upsert_v0_unstructured_v0(pdf_path):
    unstructured_vectordb.connect()
    raw_text = extract_data.extract_v0(pdf_path)
    unstructured_v0_dataset = structure_data.none_v0(raw_text, pdf_path)
    save_preprocessed_data('unstructured_data', raw_text, pdf_path,
                           'extract_v0', 'v0', 'txt')
    try:
        logging.info(logging_messages.upserting_chunks.format(constants.unstructured_tag, pdf_path))
        embedder.encode_upsert_vectordb(unstructured_v0_dataset, 10, unstructured_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.unstructured_tag, pdf_path, e))


def upsert_v0_structured_json_v1(pdf_path):
    json_structured_vectordb.connect()
    node_data = extract_data.extract_v1(pdf_path)
    json_structured_dataset, json_string = structure_data.json_v0(node_data, pdf_path)
    save_preprocessed_data('structured_data', json_string, pdf_path,
                           'extract_v1', 'v0', 'json')
    try:
        logging.info(logging_messages.upserting_chunks.format(constants.json_structured_tag, pdf_path))
        embedder.encode_upsert_vectordb(json_structured_dataset, 10, json_structured_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_all_data():
    logging.info(logging_messages.main_upserting_datasets)
    logging.info(logging_messages.main_divider)

    for pdf_path in strings.unstructured_pdf_paths:
        logging.info(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_v0_unstructured_v0(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_v0_structured_json_v1(pdf_path)

        logging.info(logging_messages.main_divider)


def main():
    upsert_all_data()

    question = "What is the section that states the limitation period for a continuous adverse possession in WA?"
    ref_answer = ("The limitation period for a continuous adverse possession in Western Australia is stated in Section "
                  "19, subsection 1 of the Limitation Act 2005. This section specifies a 12-year limitation period "
                  "for actions to recover land from the time the right to recover the land accrues, reflecting the "
                  "period during which continuous adverse possession must be maintained to claim ownership.")

    results = generate_responses(question, ref_answer)

    create_results_dataframe(results, 'question_01', 'extract_v1')

    return


if __name__ == '__main__':
    main()
