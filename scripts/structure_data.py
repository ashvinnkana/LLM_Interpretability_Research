import logging
import json

from utils import logging_messages

from utils.functions import get_file_name, save_preprocessed_data, convert_to_html, convert_to_markdown
from utils.functions import chunk_by_word_limit, build_dataset, build_dataset_v1
from utils.functions import get_node_dict, get_html_node_string
from utils.functions import get_node_dict_v2, build_dataset_v2


def none_v0(data, pdf_path):
    """
    NO STRUCTURING
    only chunking the data by word limit
    """
    logging.info(logging_messages.display_no_structuring.format('0'))
    file_name = get_file_name(pdf_path)
    chunks = chunk_by_word_limit(data)
    return build_dataset(chunks, file_name, 'UNSTRUCT', is_html=False)


def json_v0(data, pdf_path):
    """
    STRUCTURE : JSON
    Chunking all heading that has contents
    """
    logging.info(logging_messages.display_structuring_version.format('JSON', '0'))
    file_name = get_file_name(pdf_path)
    json_dict = {}
    for node in data:
        json_dict = {**json_dict, **get_node_dict(node)}

    json_string = json.dumps(json_dict, indent=2)

    return build_dataset_v1(json_dict, file_name, 'JSON', is_dict=True), json_string


def json_v1(data, pdf_path, embedder):
    """
    STRUCTURE : JSON
    Chunking all based on 2 levels that has dict
    """
    logging.info(logging_messages.display_structuring_version.format('JSON', '1'))
    file_name = get_file_name(pdf_path)

    json_dict = {}
    for index, node in enumerate(data):
        json_dict = {**json_dict, **get_node_dict_v2(node, index+1)}

    json_string = json.dumps(json_dict, indent=2)

    save_preprocessed_data('structured_data', json_string, pdf_path,
                           'extract_v2', 'v1', 'json')

    html_string = convert_to_html(json_dict)

    save_preprocessed_data('structured_data', html_string, pdf_path,
                           'extract_v2', 'v1', 'html')

    md_string = convert_to_markdown(json_dict)

    save_preprocessed_data('structured_data', md_string, pdf_path,
                           'extract_v2', 'v1', 'md')

    return build_dataset_v2(json_dict, file_name, embedder, 'JSON')


def html_v0(data, pdf_path):
    """
        STRUCTURE : HTML
        Chunking all heading that has contents
        """
    logging.info(logging_messages.display_structuring_version.format('HTML', '0'))
    file_name = get_file_name(pdf_path)
    html_format = []

    for node in data:
        html_format.append(get_html_node_string(node, 1))

    html_string = f'<html><body>\n{'\n'.join(html_format)}\n</body></html>'

    return build_dataset(html_format, file_name, 'HTML', is_html=True), html_string
