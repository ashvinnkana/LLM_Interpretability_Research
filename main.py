import json
import logging.config

import pandas

from utils import constants
from utils import logging_messages
from scripts import extract_data
from models.embedding_model import EMBEDDER
from utils.formats import v0_extraction_vectordb, v1_extraction_vectordb, v2_extraction_vectordb, format_lists
from utils.functions import download_nltk_resources, save_sample_docs, clean_for_embeds
from utils.functions import get_rouge_scores, create_results_dataframe, visualize_rouge_results
from utils import strings

# setup
download_nltk_resources()
logging.config.fileConfig(strings.logging_config_file)

embedder = EMBEDDER(constants.embedder)

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

    quest_format_r1_scores = {}
    quest_format_rL_scores = {}
    for quest in strings.questions:

        # get docs
        logging.info(logging_messages.fetching_docs.format(quest['id']))
        query_embeds = embedder.encode(clean_for_embeds(quest['query']))
        docs = {}
        for format_ in format_lists:
            format_['vector_db'].connect()
            unformatted_docs = format_['vector_db'].get_docs(quest['query'], query_embeds, format_['doc_count'])
            docs[format_['id']] = format_['get_docs_func'](unformatted_docs)
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
                        response = response.replace('\n', '\\n')  # Replace newlines with '\n'
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
        rouge1_df, rouge1_dict = create_results_dataframe(llm_format_scores, 'rouge1')
        print(f'Rouge 1 Metric - {quest['id']} Results:\n{rouge1_df}\n')
        rouge1_df.to_csv(strings.csv_results_path.format(quest['id'], 'rouge1'))
        logging.info(logging_messages.main_divider)

        rougel_df, rougel_dict = create_results_dataframe(llm_format_scores, 'rougeL')
        print(f'Rouge L Metric - {quest['id']} Results:\n{rougel_df}\n')
        rougel_df.to_csv(strings.csv_results_path.format(quest['id'], 'rougeL'))
        logging.info(logging_messages.main_divider)

        # Visualize Results
        # visualize_rouge_results(rouge1_df, rougel_df, format_lists, quest['id'])

        # fetch overall results for quest
        quest_format_r1_scores[quest['id']] = {}
        quest_format_rL_scores[quest['id']] = {}
        for format_ in format_lists:
            quest_format_r1_scores[quest['id']][format_['id']] = sum(rouge1_dict[format_['id']]) / 3
            quest_format_rL_scores[quest['id']][format_['id']] = sum(rougel_dict[format_['id']]) / 3

    # Save Overall Results
    # Convert to DataFrame and transpose
    quest_r1_df = pandas.DataFrame.from_dict(quest_format_r1_scores, orient='index').transpose()
    print(f'Rouge 1 Metric - Overall Results:\n{quest_r1_df}\n')
    quest_r1_df.to_csv(strings.csv_results_path.format('overall', 'rouge1'))

    quest_rl_df = pandas.DataFrame.from_dict(quest_format_rL_scores, orient='index').transpose()
    print(f'Rouge L Metric - Overall Results:\n{quest_rl_df}\n')
    quest_rl_df.to_csv(strings.csv_results_path.format('overall', 'rougeL'))

    # Visualize Results
    visualize_rouge_results(quest_r1_df, quest_rl_df, format_lists, 'Overall')

if __name__ == '__main__':
    main()
