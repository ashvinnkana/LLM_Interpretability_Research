import json
import logging.config

from utils import constants
from utils import logging_messages
from scripts import extract_data
from models.pipeline_llm_handler import LLM
from models.embedding_model import EMBEDDER
from utils.functions import download_nltk_resources, save_sample_docs, clean_for_embeds
from utils.functions import get_rouge_scores, create_results_dataframe, visualize_rouge_results
from utils import strings
from models.pinecone_client import PINECONE
from utils.functions import v1_json_process_docs, v2_json_process_docs, v2_markdown_process_docs
from utils.functions import v1_html_process_docs, v2_html_process_docs, unstruct_process_docs
from utils.functions import v2_toml_process_docs, v1_custom_process_docs, v2_custom_process_docs

# setup
download_nltk_resources()
logging.config.fileConfig(strings.logging_config_file)

embedder = EMBEDDER(constants.embedder)

v0_extraction_vectordb = PINECONE(strings.extraction_v0_index)
v1_extraction_vectordb = PINECONE(strings.extraction_v1_index)
v2_extraction_vectordb = PINECONE(strings.extraction_v2_index)

format_lists = [
    {'id': 'v0-unstruct',
     'query_str': strings.unstructured_question,
     'llm_msg_str': strings.unstructured_llm_message,
     'get_docs_func': unstruct_process_docs,
     'vector_db': v0_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v1-json',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v1_json_process_docs,
     'vector_db': v1_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v1-html',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v1_html_process_docs,
     'vector_db': v1_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-toml',
     'query_str': strings.toml_question,
     'llm_msg_str': strings.toml_llm_message,
     'get_docs_func': v2_toml_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 2},
    {'id': 'v2-md',
     'query_str': strings.md_question,
     'llm_msg_str': strings.md_llm_message,
     'get_docs_func': v2_markdown_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-json',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v2_json_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-html',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v2_html_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-custom1',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v1_custom_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-custom2',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v2_custom_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4}
]


def legal_llm_response(question, docs):
    legal_llm = LLM()
    legal_llm.set_model(constants.aus_legal_llm)
    print(f'{constants.aus_legal_llm}\n{legal_llm.generate_response(question, docs)}\n')


def upsert_extract_v0(pdf_path):
    v0_extraction_vectordb.connect()
    dataset = extract_data.extract_v0(pdf_path, embedder)
    try:
        logging.info(logging_messages.upserting_chunks.format('EXTRACT-V0', pdf_path))
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, v0_extraction_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_extract_v1(pdf_path):
    v1_extraction_vectordb.connect()
    dataset = extract_data.extract_v1(pdf_path, embedder)
    try:
        logging.info(logging_messages.upserting_chunks.format('EXTRACT-V1', pdf_path))
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, v1_extraction_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_extract_v2(pdf_path):
    v2_extraction_vectordb.connect()
    dataset = extract_data.extract_v2(pdf_path, embedder)

    try:
        logging.info(logging_messages.upserting_chunks.format('EXTRACT-V2', pdf_path))
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, v2_extraction_vectordb)
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_all_data():
    logging.info(logging_messages.main_upserting_datasets)
    logging.info(logging_messages.main_divider)

    for pdf_path in strings.unstructured_pdf_paths:
        logging.info(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_extract_v0(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_extract_v1(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_extract_v2(pdf_path)

        logging.info(logging_messages.main_divider)


def main():
    print("RUNNING ON: ", embedder.get_encoder_device())
    upsert_all_data()

    logging.info(logging_messages.main_divider)
    topic = 'LAW'
    for quest in strings.questions:

        # get docs
        logging.info(logging_messages.fetching_docs.format(quest['id']))
        query_embeds = embedder.encode(clean_for_embeds(quest['query']))
        docs = {}
        for format_ in format_lists:
            format_['vector_db'].connect()
            docs[format_['id']] = format_['get_docs_func'](format_['vector_db']
                                                           .get_docs(quest['query'], query_embeds, format_['doc_count']))
            logging.info(format_['id'] + ' : ' + str(embedder.count_tokens(docs[format_['id']])))
        save_sample_docs(json.dumps(docs), quest['id'])
        logging.info(logging_messages.main_divider)
        llm_format_scores = {}
        # generate responses
        for llm in constants.non_legal_llm_list:
            logging.info(f'>> Generating Responses using {llm['model_id']}')
            llm['client'].set_model(llm['model_id'])
            logging.info(logging_messages.sub_divider)
            scores = {}
            for i in range(3):
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

                logging.info(scores)
                logging.info(logging_messages.main_divider)
            llm_format_scores[llm['model_id']] = scores
            logging.info(logging_messages.main_divider)

        # Save Results
        rouge1_df = create_results_dataframe(llm_format_scores, 'rouge1')
        print(f'Rouge 1 Metric - {quest['id']} Results:\n{rouge1_df}\n')
        rouge1_df.to_csv(strings.csv_results_path.format(quest['id'], 'rouge1'))
        logging.info(logging_messages.main_divider)

        rougel_df = create_results_dataframe(llm_format_scores, 'rougeL')
        print(f'Rouge L Metric - {quest['id']} Results:\n{rougel_df}\n')
        rougel_df.to_csv(strings.csv_results_path.format(quest['id'], 'rougeL'))
        logging.info(logging_messages.main_divider)

        # Visualize Results
        visualize_rouge_results(rouge1_df, rougel_df, format_lists, quest['id'])


if __name__ == '__main__':
    main()
