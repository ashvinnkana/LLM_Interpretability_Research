import json
import logging.config

import pandas
from datasets import Dataset

from utils import constants
from utils import logging_messages
from scripts import extract_data
from models.embedding_model import EMBEDDER
from utils.formats import v0_extraction_vectordb, v1_extraction_vectordb, v2_extraction_vectordb, format_lists, \
    v2_1_extraction_vectordb, unstruct_io_vectordb
from utils.functions import download_nltk_resources, save_sample_docs, clean_for_embeds, get_docs_v2_1, \
    get_overall_scores, get_overall_format_order, save_checkpoint, re_arrange_dict, create_prompt_for_legal_llm, \
    merge_chunks_v2_1
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
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, v0_extraction_vectordb, 'v0')
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_extract_v1(pdf_path):
    v1_extraction_vectordb.connect()
    dataset = extract_data.extract_v1(pdf_path, embedder)
    try:
        logging.info(logging_messages.upserting_chunks.format('EXTRACT-V1', pdf_path))
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, v1_extraction_vectordb, 'v1')
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_extract_v2(pdf_path):
    v2_extraction_vectordb.connect()
    dataset = extract_data.extract_v2(pdf_path, embedder)

    try:
        logging.info(logging_messages.upserting_chunks.format('EXTRACT-V2', pdf_path))
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, v2_extraction_vectordb, 'v2')
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_extract_v2_1(pdf_path):
    v2_1_extraction_vectordb.connect()
    dataset = extract_data.extract_v2_1(pdf_path, embedder)

    try:
        logging.info(logging_messages.upserting_chunks.format('EXTRACT-V2.1', pdf_path))
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, v2_1_extraction_vectordb, 'v2.1')
        logging.info(logging_messages.status_success)
    except Exception as e:
        logging.error(logging_messages.error_upserting.format(constants.json_structured_tag, pdf_path, e))


def upsert_unstructured_io_extract(pdf_path):
    unstruct_io_vectordb.connect()

    file_name = pdf_path.split('/')[-1].split('.')[0]
    with open(f'data/chunks/by_unstructured_io/{file_name}_chunks.json', 'r') as json_file:
        data = json.load(json_file)

    dataset = Dataset.from_list(data)

    try:
        logging.info(logging_messages.upserting_chunks.format('UNSTRUCT-IO', pdf_path))
        embedder.encode_upsert_vectordb(dataset, constants.upsert_batch_size, unstruct_io_vectordb, 'unstruct-io')
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

        logging.info(logging_messages.sub_divider)
        upsert_extract_v2_1(pdf_path)

        logging.info(logging_messages.sub_divider)
        upsert_unstructured_io_extract(pdf_path)

        logging.info(logging_messages.main_divider)


def main():
    print("RUNNING ON: ", embedder.get_encoder_device())
    upsert_all_data()

    logging.info(logging_messages.main_divider)
    topic = 'LEGAL'

    quest_format_r1_scores = {}
    quest_format_rL_scores = {}

    llm_order = [item['model_id'] for item in constants.non_legal_llm_list[:3]]

    for quest in strings.questions:

        # get docs
        logging.info(logging_messages.fetching_docs.format(quest['id']))
        query_embeds = embedder.encode(clean_for_embeds(quest['query']))
        docs = {}

        v2_1_extraction_vectordb.connect()
        v2_1_dicts, v2_1_texts = get_docs_v2_1(v2_1_extraction_vectordb.get_docs(quest['query'], query_embeds, 3),
                                               embedder)

        v2_1_merged_docs_3 = merge_chunks_v2_1(v2_1_dicts, quest['query'], 3)

        for format_ in format_lists:
            if format_['version'] == 2.1:
                docs[format_['id']] = format_['get_docs_func'](v2_1_merged_docs_3)
            else:
                format_['vector_db'].connect()
                unformatted_docs = format_['vector_db'].get_docs(quest['query'], query_embeds, format_['doc_count'])
                docs[format_['id']] = format_['get_docs_func'](unformatted_docs)
            logging.info(format_['id'] + ' : ' + str(embedder.count_tokens(docs[format_['id']])))

        docs['prompt'] = create_prompt_for_legal_llm(quest['query'], docs['v2.1-unstruct'])
        save_sample_docs(json.dumps(docs), quest['id'])

        logging.info(logging_messages.main_divider)
        llm_format_scores = {}
        # generate responses
        for llm in constants.non_legal_llm_list[:3]:
            logging.info(f'>> Generating Responses using {llm['model_id']}')
            llm['client'].set_model(llm['model_id'])
            logging.info(logging_messages.sub_divider)
            scores = {}
            for i in range(1):
                logging.info(f'- Attempt 0{i + 1}')
                for format_ in format_lists:
                    query = format_['query_str'].format(quest['query'])
                    system_message = format_['llm_msg_str'].format(topic, docs[format_['id']])
                    response = ''

                    check = True
                    while check:
                        try:
                            response = llm['client'].generate_response(query, system_message)

                            response = response.replace('\n', '\\n')  # Replace newlines with '\n'
                            logging.info(f'-- {format_['id']} :: {response}')
                            check = False
                        except Exception as e:
                            logging.error(f'-- {format_['id']} :: FAILED({e})')
                            error_message = e.args[0]
                            if isinstance(error_message, dict) and 'error' in error_message and 'type' in error_message['error']:
                                if 'internal_server_error' != error_message['error']['type']:
                                    check = False
                                else:
                                    logging.info(f'-- RETRYING ...')
                            else:
                                check = False

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
        visualize_rouge_results(rouge1_df, rougel_df, [item['id'] for item in format_lists], quest['id'])

        # fetch overall results for quest
        quest_format_r1_scores[quest['id']] = {}
        quest_format_rL_scores[quest['id']] = {}
        for format_ in format_lists:
            for index, llm in enumerate(llm_order):
                if format_['id'] in quest_format_r1_scores[quest['id']]:
                    quest_format_r1_scores[quest['id']][format_['id']][llm] = rouge1_dict[format_['id']][index]
                    quest_format_rL_scores[quest['id']][format_['id']][llm] = rougel_dict[format_['id']][index]
                else:
                    quest_format_r1_scores[quest['id']][format_['id']] = {llm: rouge1_dict[format_['id']][index]}
                    quest_format_rL_scores[quest['id']][format_['id']] = {llm: rougel_dict[format_['id']][index]}

        save_checkpoint(quest_format_r1_scores, quest['id'], 'rouge1')
        save_checkpoint(quest_format_rL_scores, quest['id'], 'rougeL')

    # Save Question Results
    save_checkpoint(quest_format_r1_scores, 'questions', 'rouge1')
    save_checkpoint(quest_format_rL_scores, 'questions', 'rougeL')

    #with open('results/temp/questions_rouge1_checkpoint.json', 'r') as json_file:
    #    quest_format_r1_scores = json.load(json_file)

    #with open('results/temp/questions_rougeL_checkpoint.json', 'r') as json_file:
    #    quest_format_rL_scores = json.load(json_file)

    quest_format_r1_scores = re_arrange_dict(quest_format_r1_scores)
    quest_format_rL_scores = re_arrange_dict(quest_format_rL_scores)

    for llm in llm_order:
        # Convert to DataFrame and transpose
        quest_r1_df = pandas.DataFrame.from_dict(quest_format_r1_scores[llm], orient='index').transpose()
        print(f'Rouge 1 Metric - {llm} Questions Results:\n{quest_r1_df}\n')
        quest_r1_df.to_csv(strings.csv_results_path.format(f'questions-{llm}', 'rouge1'))

        quest_rl_df = pandas.DataFrame.from_dict(quest_format_rL_scores[llm], orient='index').transpose()
        print(f'Rouge L Metric - {llm} Questions Results:\n{quest_rl_df}\n')
        quest_rl_df.to_csv(strings.csv_results_path.format(f'questions-{llm}', 'rougeL'))

        # Visualize Results
        visualize_rouge_results(quest_r1_df, quest_rl_df, [item['id'] for item in format_lists], f'questions-{llm}')

    logging.info(logging_messages.main_divider)

    # Save LLM Results
    llm_format_r1_scores = {}
    llm_format_rL_scores = {}

    for llm in llm_order:
        llm_format_r1_scores[llm] = {}
        llm_format_rL_scores[llm] = {}

    for llm in llm_order:
        for quest in strings.questions:
            for format_ in format_lists:
                if format_['id'] in llm_format_r1_scores[llm]:
                    llm_format_r1_scores[llm][format_['id']]['total'] += quest_format_r1_scores[llm][quest['id']][format_['id']]
                    llm_format_r1_scores[llm][format_['id']]['count'] += 1
                    llm_format_rL_scores[llm][format_['id']]['total'] += quest_format_rL_scores[llm][quest['id']][format_['id']]
                    llm_format_rL_scores[llm][format_['id']]['count'] += 1
                else:
                    llm_format_r1_scores[llm][format_['id']] = {'total': quest_format_r1_scores[llm][quest['id']][format_['id']], 'count': 1}
                    llm_format_rL_scores[llm][format_['id']] = {'total': quest_format_rL_scores[llm][quest['id']][format_['id']], 'count': 1}

    for format_ in format_lists:
        for llm in llm_order:
            llm_format_r1_scores[llm][format_['id']] = llm_format_r1_scores[llm][format_['id']]['total'] / \
                                                       llm_format_r1_scores[llm][format_['id']]['count']
            llm_format_rL_scores[llm][format_['id']] = llm_format_rL_scores[llm][format_['id']]['total'] / \
                                                       llm_format_rL_scores[llm][format_['id']]['count']


    # Convert to DataFrame and transpose
    llm_r1_df = pandas.DataFrame.from_dict(llm_format_r1_scores, orient='index').transpose()
    print(f'Rouge 1 Metric - LLM Results:\n{llm_r1_df}\n')
    llm_r1_df.to_csv(strings.csv_results_path.format('llms', 'rouge1'))

    llm_rl_df = pandas.DataFrame.from_dict(llm_format_rL_scores, orient='index').transpose()
    print(f'Rouge L Metric - LLM Results:\n{llm_rl_df}\n')
    llm_rl_df.to_csv(strings.csv_results_path.format('llms', 'rougeL'))

    # Visualize Results
    visualize_rouge_results(llm_r1_df, llm_rl_df, [item['id'] for item in format_lists], 'llms')

    logging.info(logging_messages.main_divider)

    # Save Overall Results
    # Initialize the 'overall' dictionary
    r1_overall_scores = get_overall_scores(quest_format_r1_scores)
    rl_overall_scores = get_overall_scores(quest_format_rL_scores)

    # Convert to DataFrame and transpose
    overall_r1_df = pandas.DataFrame.from_dict(r1_overall_scores, orient='index').transpose()
    print(f'Rouge 1 Metric - Overall Results:\n{overall_r1_df}\n')
    overall_r1_df.to_csv(strings.csv_results_path.format('overall', 'rouge1'))

    overall_rl_df = pandas.DataFrame.from_dict(rl_overall_scores, orient='index').transpose()
    print(f'Rouge L Metric - Overall Results:\n{overall_rl_df}\n')
    overall_rl_df.to_csv(strings.csv_results_path.format('overall', 'rougeL'))

    # Visualize Results
    format_order = get_overall_format_order(r1_overall_scores, rl_overall_scores)
    visualize_rouge_results(overall_r1_df, overall_rl_df, format_order, 'overall')


if __name__ == '__main__':
    main()
