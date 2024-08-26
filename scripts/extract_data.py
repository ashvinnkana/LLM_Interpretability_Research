import re
import logging
import itertools

from utils import logging_messages, constants, regex_patterns

from models.node import Node
from utils.functions import extract_pdf_raw_text, get_file_name, get_node_dict_v2, \
    build_dataset_v2, save_preprocessed_data, build_dataset_v0, get_node_dict, build_dataset_v1, \
    save_all_format_structuring_v2, save_all_format_structuring_v1
from utils.functions import clean
from utils.functions import extract_headers_and_footers, remove_header_footer
from utils.functions import group_sentences
from utils.functions import is_heading, is_bullet
from utils.functions import process_heading, process_ordered_bullet, process_unordered_bullet, process_context
from utils.functions import get_level_path, add_node_child
from utils.functions import extract_pdf_metadata, classify_page_text_by_levels
from utils.functions import extract_headers_and_footers_v2, remove_header_footer_v2
from utils.functions import clean_text_by_types_v2, clean_text_by_formats_v2, classify_page_text_by_types


def extract_v0(pdf_path, embedder):
    logging.info(logging_messages.display_extraction_version.format('0'))
    file_name = get_file_name(pdf_path)

    _, extracted_text = extract_pdf_raw_text(pdf_path)
    cleaned_text = clean(extracted_text)

    save_preprocessed_data('unstructured_data', cleaned_text, pdf_path,
                           'extract_v0', 'UNSTRUCT', 'txt')

    return build_dataset_v0(cleaned_text.split('\n'), file_name, embedder, 'extract_v0')


def extract_v1(pdf_path, embedder):
    logging.info(logging_messages.display_extraction_version.format('1'))
    file_name = get_file_name(pdf_path)

    pages, _ = extract_pdf_raw_text(pdf_path)

    headers, footers = extract_headers_and_footers(pages)
    contents_by_pages = remove_header_footer(pages, headers, footers)

    content_phrases = list(itertools.chain.from_iterable(contents_by_pages))
    content_sentences = group_sentences(content_phrases)

    extracted_data = []
    latest_head_index = 0
    node_route = []
    bullet_root_indexes = {}

    logging.info(logging_messages.classify_sentences)
    for index, sentence in enumerate(content_sentences):
        if is_heading(sentence):
            node_route, latest_head_index = process_heading(extracted_data, sentence, index, node_route)
            continue

        bullet, value = is_bullet(sentence)
        if bullet is not None:
            if bullet != '*':
                node_route, bullet_root_indexes = (
                    process_ordered_bullet(extracted_data, sentence, bullet, value,
                                           node_route, bullet_root_indexes, latest_head_index))
            else:
                node_route = process_unordered_bullet(extracted_data, sentence, value, index, node_route)
        else:
            node_route = process_context(extracted_data, sentence, index, node_route)

    json_dict = {}
    for node in extracted_data:
        json_dict = {**json_dict, **get_node_dict(node)}

    save_all_format_structuring_v1(json_dict, pdf_path, 'extract_v1')

    return build_dataset_v1(json_dict, file_name, embedder, 'extract_v1')


def extract_v2(pdf_path, embedder):
    logging.info(logging_messages.display_extraction_version.format('2'))
    file_name = get_file_name(pdf_path)

    pages = extract_pdf_metadata(pdf_path)

    lvl_classified_pages = classify_page_text_by_levels(pages)

    header_footer_levels = extract_headers_and_footers_v2(lvl_classified_pages)
    content_pages = remove_header_footer_v2(lvl_classified_pages, header_footer_levels)

    cleaned_pages = clean_text_by_formats_v2(content_pages)
    type_classified_pages = classify_page_text_by_types(cleaned_pages)
    extractable_data = clean_text_by_types_v2(type_classified_pages)

    extracted_data = [constants.root_node_v2]
    current_level_path = []
    current_node_route = [0]
    node_routes_history = {}
    for index, span in enumerate(extractable_data):
        node = Node(span['level'], re.sub(regex_patterns.no_special_characters_v2, '',
                                          span['text'].strip()), span['type_index'])
        if f"|{span['level']}.{span['type_index']}" in current_level_path:
            current_level_path = get_level_path(f"|{span['level']}.{span['type_index']}", current_level_path)
            current_node_route = node_routes_history[''.join(current_level_path)].copy()
            current_node_route.append(add_node_child(extracted_data, current_node_route, node))
        else:
            current_level_path.append(f"|{span['level']}.{span['type_index']}")
            node_routes_history[''.join(current_level_path)] = current_node_route.copy()
            current_node_route.append(add_node_child(extracted_data, current_node_route, node))

    json_dict = {}
    for index, node in enumerate(extracted_data):
        json_dict = {**json_dict, **get_node_dict_v2(node, index + 1)}

    save_all_format_structuring_v2(json_dict['structured_data'], pdf_path, 'extract_v2')

    return build_dataset_v2(json_dict, file_name, embedder, 'extract_v2')
