import logging
import json

from utils import logging_messages

from utils.functions import get_file_name
from utils.functions import chunk_by_word_limit, build_dataset
from utils.functions import get_node_dict


def none_v0(data, pdf_path):
    """
    NO STRUCTURING
    only chunking the data by word limit
    """
    logging.info(logging_messages.display_no_structuring.format('0'))
    file_name = get_file_name(pdf_path)
    chunks = chunk_by_word_limit(data)
    return build_dataset(chunks, file_name, is_dict=False)


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

    return build_dataset(json_dict, file_name, is_dict=True), json_string
