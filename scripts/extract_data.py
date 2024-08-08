import logging
import itertools

from utils import logging_messages

from utils.functions import extract_pdf_raw_text
from utils.functions import clean
from utils.functions import extract_headers_and_footers, remove_header_footer
from utils.functions import group_sentences
from utils.functions import is_heading, is_bullet
from utils.functions import process_heading, process_ordered_bullet, process_unordered_bullet, process_context


def extract_v0(pdf_path):
    """extract raw text with some basic clean"""
    logging.info(logging_messages.display_extraction_version.format('0'))
    _, extracted_text = extract_pdf_raw_text(pdf_path)
    return clean(extracted_text)


def extract_v1(pdf_path):
    """
    extract text Nodes by headings, ordered and unordered lists & content blocks
    - remove headers and footers
    - group phrases to make sentences
    - process headings, bullets and contents
    - hierarchical node object list
    """
    logging.info(logging_messages.display_extraction_version.format('1'))
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

    return extracted_data
