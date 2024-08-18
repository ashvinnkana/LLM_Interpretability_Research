import logging.config

from utils import constants
from utils import logging_messages
from scripts import extract_data, structure_data
from models.pipeline_llm_handler import LLM
from models.embedding_model import EMBEDDER
from utils.functions import download_nltk_resources
from utils.functions import save_preprocessed_data
from utils.functions import get_rouge_scores, create_results_dataframe, visualize_rouge_results
from utils import strings
from models.pinecone_client import PINECONE
from utils.functions import v1_json_process_docs, v2_json_process_docs
from utils.functions import v1_html_process_docs, v2_html_process_docs, unstruct_process_docs

# setup
download_nltk_resources()
logging.config.fileConfig(strings.logging_config_file)

embedder = EMBEDDER(constants.embedder)

unstructured_vectordb = PINECONE(strings.unstructured_0_index)
v1_json_structured_vectordb = PINECONE(strings.structured_JSON_1_index)
v1_html_structured_vectordb = PINECONE(strings.structured_HTML_1_index)
v2_structured_vectordb = PINECONE(strings.structured_v2_index)

format_lists = [
    {'id': 'unstructured',
     'query_str': strings.unstructured_question,
     'llm_msg_str': strings.unstructured_llm_message,
     'get_docs_func': unstruct_process_docs,
     'vector_db': unstructured_vectordb},
    {'id': 'json-structured-v1',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v1_json_process_docs,
     'vector_db': v1_json_structured_vectordb},
    {'id': 'html-structured-v1',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v1_html_process_docs,
     'vector_db': v1_html_structured_vectordb},
    {'id': 'json-structured-v2',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v2_json_process_docs,
     'vector_db': v2_structured_vectordb},
    {'id': 'html-structured-v2',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v2_html_process_docs,
     'vector_db': v2_structured_vectordb}
]


def legal_llm_response(question, docs):
    legal_llm = LLM(constants.aus_legal_llm, 'question-answering')
    print(f'{constants.aus_legal_llm}\n{legal_llm.generate(question, docs)}\n')


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


def upsert_extract_v2(pdf_path):
    v2_structured_vectordb.connect()
    node_data = extract_data.extract_v2(pdf_path)
    json_structured_dataset, json_string = structure_data.json_v1(node_data, pdf_path, embedder)
    save_preprocessed_data('structured_data', json_string, pdf_path,
                           'extract_v2', 'v1', 'json')
    try:
        logging.info(logging_messages.upserting_chunks.format(constants.json_structured_tag, pdf_path))
        embedder.encode_upsert_vectordb(json_structured_dataset, 5, v2_structured_vectordb)
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
        upsert_extract_v2(pdf_path)

        logging.info(logging_messages.main_divider)


def main():
    print("RUNNING ON: ", embedder.get_encoder_device())
    # upsert_all_data()

    logging.info(logging_messages.main_divider)
    topic = 'LAW'
    for quest in strings.questions:

        # get docs
        logging.info(logging_messages.fetching_docs.format(quest['id']))
        query_embeds = embedder.encode(quest['query'])
        docs = {}
        for format_ in format_lists:
            format_['vector_db'].connect()
            docs[format_['id']] = format_['get_docs_func'](format_['vector_db']
                                                           .get_docs(query_embeds, constants.fetch_docs_count))
        logging.info(logging_messages.main_divider)
        llm_format_scores = {}
        # generate responses
        for llm in constants.non_legal_llm_list:
            logging.info(f'>> Generating Responses using {llm['model_id']}')
            llm['client'].set_model(llm['model_id'])
            logging.info(logging_messages.sub_divider)
            scores = {}
            for i in range(2):
                logging.info(f'- Attempt 0{i + 1}')
                for format_ in format_lists:
                    query = format_['query_str'].format(quest['query'])
                    system_message = format_['llm_msg_str'].format(topic, docs[format_['id']])
                    response = ''
                    try:
                        response = llm['client'].generate_response(query, system_message)
                        logging.info(f'-- {format_['id']} :: {response}')
                    except Exception as e:
                        logging.error(f'-- {format_['id']} :: FAILED({e})')

                    # record scores
                    score = get_rouge_scores(quest['ref_answer'], response)
                    try:
                        scores[format_['id']].append(score)
                    except KeyError:
                        scores[format_['id']] = []
                        scores[format_['id']].append(score)
                logging.info(logging_messages.main_divider)
            llm_format_scores[llm['model_id']] = scores
            logging.info(logging_messages.main_divider)

        # Save Results
        rouge1_df = create_results_dataframe(llm_format_scores, 'rouge1')
        print(f'Rouge 1 Metric - {quest['id']} Results:\n{rouge1_df}\n')
        rouge1_df.to_csv(strings.csv_results_path.format(quest['id'], 'rouge1'))
        logging.info(logging_messages.main_divider)

        rougeL_df = create_results_dataframe(llm_format_scores, 'rougeL')
        print(f'Rouge L Metric - {quest['id']} Results:\n{rougeL_df}\n')
        rougeL_df.to_csv(strings.csv_results_path.format(quest['id'], 'rougeL'))
        logging.info(logging_messages.main_divider)

        # Visualize Results
        visualize_rouge_results(rouge1_df, rougeL_df, format_lists, quest['id'])


if __name__ == '__main__':
    main()
