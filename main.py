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

v1_json_structured_vectordb = PINECONE(strings.structured_JSON_1_index)
v2_json_structured_vectordb = PINECONE(strings.structured_JSON_2_index)

v1_html_structured_vectordb = PINECONE(strings.structured_HTML_1_index)


def legal_llm_response(question, docs):
    legal_llm = LLM(constants.aus_legal_llm, 'question-answering')
    print(f'{constants.aus_legal_llm}\n{legal_llm.generate(question, docs)}\n')


def create_results_dataframe(results, question_id):
    logging.info(logging_messages.saving_responses)
    # Extract model names
    model_names = [model['model_id'] for model in constants.groq_supported_llm_list]

    # Prepare data dynamically
    data_rouge1 = []
    data_rougeL = []
    for key in results[0].keys():
        avg_rouge1 = []
        avg_rougeL = []
        for idx in range(len(constants.groq_supported_llm_list)):
            try:
                rouge1_scores = [results[i][key][idx]['rouge1'] for i in range(len(results))]
                rougeL_scores = [results[i][key][idx]['rougeL'] for i in range(len(results))]

                avg_rouge1.append(sum(rouge1_scores) / len(rouge1_scores))
                avg_rougeL.append(sum(rougeL_scores) / len(rougeL_scores))
            except TypeError as e:
                logging.error(logging_messages.error_saving_response.format(e))
                avg_rouge1.append(None)
                avg_rougeL.append(None)
                continue
        data_rouge1.append(avg_rouge1)
        data_rougeL.append(avg_rougeL)

    # Index for the rows
    index = list(results[0].keys())

    # Create DataFrame for rouge1
    df_rouge1 = pd.DataFrame(data_rouge1, index=index, columns=model_names)
    print(f'Rouge 1 Metric Results:\n{df_rouge1}\n')
    df_rouge1.to_csv(strings.csv_results_path.format(question_id, 'rouge1'))

    # Create DataFrame for rougeL
    df_rougeL = pd.DataFrame(data_rougeL, index=index, columns=model_names)
    print(f'Rouge L Metric Results:\n{df_rougeL}\n')
    df_rougeL.to_csv(strings.csv_results_path.format(question_id, 'rougeL'))


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
    logging.info(logging_messages.generating_response.format('JSON-STRUCTURED-V1'))
    v1_json_structured_vectordb.connect()
    json_structured_docs = v1_json_structured_vectordb.get_docs(query_embeds, 5)
    response_scores['json-structured-v1'] = non_legal_llm_responses(question, json_structured_docs, ref_answer)

    logging.info(logging_messages.generating_response.format('JSON-STRUCTURED-V2'))
    v2_json_structured_vectordb.connect()
    json_structured_docs = v2_json_structured_vectordb.get_docs(query_embeds, 5)
    response_scores['json-structured-v2'] = non_legal_llm_responses(question, json_structured_docs, ref_answer)

    # get responses for html_structured data
    logging.info(logging_messages.generating_response.format('HTML-STRUCTURED-V1'))
    v1_html_structured_vectordb.connect()
    html_structured_docs = v1_html_structured_vectordb.get_docs(query_embeds, 5)
    response_scores['html-structured-v1'] = non_legal_llm_responses(question, html_structured_docs, ref_answer)

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


def upsert_v1_structured_json_v2(pdf_path):
    v2_json_structured_vectordb.connect()
    node_data = extract_data.extract_v2(pdf_path)
    json_structured_dataset, json_string = structure_data.json_v1(node_data, pdf_path)
    save_preprocessed_data('structured_data', json_string, pdf_path,
                           'extract_v2', 'v1', 'json')
    try:
        logging.info(logging_messages.upserting_chunks.format(constants.json_structured_tag, pdf_path))
        embedder.encode_upsert_vectordb(json_structured_dataset, 5, v2_json_structured_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_v0_structured_json_v1(pdf_path):
    v1_json_structured_vectordb.connect()
    node_data = extract_data.extract_v1(pdf_path)
    json_structured_dataset, json_string = structure_data.json_v0(node_data, pdf_path)
    save_preprocessed_data('structured_data', json_string, pdf_path,
                           'extract_v1', 'v0', 'json')
    try:
        logging.info(logging_messages.upserting_chunks.format(constants.json_structured_tag, pdf_path))
        embedder.encode_upsert_vectordb(json_structured_dataset, 5, v1_json_structured_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_v0_structured_html_v1(pdf_path):
    v1_html_structured_vectordb.connect()
    node_data = extract_data.extract_v1(pdf_path)
    html_structured_dataset, html_string = structure_data.html_v0(node_data, pdf_path)
    save_preprocessed_data('structured_data', html_string, pdf_path,
                           'extract_v1', 'v0', 'html')
    try:
        logging.info(logging_messages.upserting_chunks.format(constants.html_structured_tag, pdf_path))
        embedder.encode_upsert_vectordb(html_structured_dataset, 5, v1_html_structured_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.html_structured_tag, pdf_path, e))


def upsert_all_data():
    logging.info(logging_messages.main_upserting_datasets)
    logging.info(logging_messages.main_divider)

    for pdf_path in strings.unstructured_pdf_paths:
        logging.info(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_v0_unstructured_v0(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_v0_structured_json_v1(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_v0_structured_html_v1(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_v1_structured_json_v2(pdf_path)

        logging.info(logging_messages.main_divider)


def main():
    print("RUNNING ON: ", embedder.get_encoder_device())
    # upsert_all_data()

    question = "What is the section that states the limitation period for a continuous adverse possession in WA?"
    ref_answer = ("The limitation period for a continuous adverse possession in Western Australia is stated in Section "
                  "19, subsection 1 of the Limitation Act 2005. This section specifies a 12-year limitation period "
                  "for actions to recover land from the time the right to recover the land accrues, reflecting the "
                  "period during which continuous adverse possession must be maintained to claim ownership.")

    results = []
    for i in range(5):
        logging.info(f'GENERATED RESPONSES 0{i + 1}')
        logging.info(logging_messages.sub_divider)
        results.append(generate_responses(question, ref_answer))

    create_results_dataframe(results, 'question_01')


if __name__ == '__main__':
    main()
