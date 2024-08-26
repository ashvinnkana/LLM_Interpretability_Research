import json
import re
import copy
import nltk
import logging
import pdfplumber
import string
import fitz
import ast
import math
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from collections import Counter
from utils import strings, constants, regex_patterns, logging_messages

from models.node import Node


def clean_for_embeds(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and symbols using regular expressions
    text = re.sub(regex_patterns.alpha_characters_only_simple, '', text)

    # Remove stop words
    stop_words = set(stopwords.words(constants.docs_language))
    filtered_words = [word for word in text.split() if word not in stop_words]

    # Join the cleaned words back into a string
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text


def save_preprocessed_data(type_, data, unstructured_file_path, extract_version, struct_version, file_extension):
    file_name = (unstructured_file_path.split('/')[-1]).split('.')[0]
    structured_file_path = strings.structured_data_path.format(type_, file_name, extract_version, struct_version,
                                                               file_extension)
    with open(structured_file_path, 'w') as file:
        file.write(data)


def save_all_format_structuring_v1(json_dict, pdf_path, extract_version):
    json_string = json.dumps(json_dict, indent=2)

    save_preprocessed_data('structured_data', json_string, pdf_path,
                           extract_version, 'JSON', 'json')

    html_string = convert_to_html(json_dict)

    save_preprocessed_data('structured_data', html_string, pdf_path,
                           extract_version, 'HTML', 'html')

    md_string = convert_to_markdown(json_dict)

    save_preprocessed_data('structured_data', md_string, pdf_path,
                           extract_version, 'MD', 'md')

    toml_string = convert_to_toml(json_dict, [])

    save_preprocessed_data('structured_data', toml_string, pdf_path,
                           extract_version, 'TOML', 'toml')


def save_all_format_structuring_v2(json_dict, pdf_path, extract_version):
    json_string = json.dumps(json_dict, indent=2)

    save_preprocessed_data('structured_data', json_string, pdf_path,
                           extract_version, 'JSON', 'json')

    html_string = convert_to_html(json_dict)

    save_preprocessed_data('structured_data', html_string, pdf_path,
                           extract_version, 'HTML', 'html')

    md_string = convert_to_markdown(json_dict)

    save_preprocessed_data('structured_data', md_string, pdf_path,
                           extract_version, 'MD', 'md')

    toml_string = convert_to_toml(json_dict, [])

    save_preprocessed_data('structured_data', toml_string, pdf_path,
                           extract_version, 'TOML', 'toml')

    custom_1_string = convert_to_custom_v1(json_dict)

    save_preprocessed_data('structured_data', custom_1_string, pdf_path,
                           extract_version, 'CUSTOM_V1', 'html')

    custom_2_string = convert_to_custom_v2(json_dict)

    save_preprocessed_data('structured_data', custom_2_string, pdf_path,
                           extract_version, 'CUSTOM_V2', 'json')


def save_sample_docs(docs, question_id):

    sample_docs_path = strings.samples_docs_path.format(question_id)
    with open(sample_docs_path, 'w') as file:
        file.write(docs)


def get_html_node_string(node, level):
    html_string = ""

    # Handle nodes of type 'heading'
    if node.type_ == 'heading':
        # If the node has no children, create a heading tag
        if len(node.children) == 0:
            html_string += f'<h{level}>{node.value}</h{level}>'
        else:
            # If the node has children, wrap the heading in a div and process children
            html_string += f'<div><h{level}>{node.value}</h{level}>'

            for child in node.children:
                html_string += get_html_node_string(child, level + 1)

            html_string += '</div>'

    # Handle nodes of type 'ordered_bullet'
    elif node.type_ == 'ordered_bullet':
        # If the node has no children, create an ordered list item with key
        if len(node.children) == 0:
            html_string += f'<h{level}>({node.key}) {node.value}</h{level}>'
        else:
            # If the node has children, wrap the item in a div and process children
            html_string += f'<div><h{level}>({node.key}) {node.value}</h{level}>'

            for child in node.children:
                html_string += get_html_node_string(child, level + 1)

            html_string += '</div>'

    # Handle nodes of type 'unordered_bullet'
    elif node.type_ == 'unordered_bullet':
        # If the node has no children, create an unordered list item
        if len(node.children) == 0:
            html_string += f'<li>{node.value}</li>'
        else:
            # If the node has children, wrap the item in a div and process children
            html_string += f'<div><li>{node.value}</li>'

            for child in node.children:
                html_string += get_html_node_string(child, level + 1)

            html_string += '</div>'

    # Handle nodes of type 'content'
    elif node.type_ == 'content':
        # If the node has no children, create a paragraph
        if len(node.children) == 0:
            html_string += f'<p>{node.value}</p>'
        else:
            # If the node has children, wrap the paragraph in a div and process children
            html_string += f'<div><p>{node.value}</p>'

            for child in node.children:
                html_string += get_html_node_string(child, level + 1)

            html_string += '</div>'

    return html_string


def extract_scores(score_dict):
    """
    Extract the F-measure scores from a dictionary of ROUGE scores.

    Args:
        score_dict (dict): A dictionary containing ROUGE scores, where each key corresponds
                               to a ROUGE metric (e.g., 'rouge1', 'rougeL') and each value
                               is a Score object containing precision, recall, and fmeasure.

    Returns:
        dict: A dictionary where keys are ROUGE metric names and values are the corresponding F-measure scores.
        :param score_dict:
        :param response:
    """
    dict_ = {k: v.fmeasure for k, v in score_dict.items()}
    return dict_


def get_rouge_scores(ref_answer, response):
    """
    Calculate the ROUGE scores between a reference answer and a generated response.

    Args:
        ref_answer (str): The reference text against which the response will be evaluated.

    Returns:
        dict: A dictionary containing the F-measure scores for the specified ROUGE metrics ('rouge1', 'rougeL').
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return extract_scores(scorer.score(ref_answer, response))


def get_node_dict(node):
    """
    Recursively converts a node and its children into a dictionary representation.

    Args:
    - node (Node): The node to convert.

    Returns:
    - dict: The dictionary representation of the node and its children.
    """
    children_node = {}
    # Recursively process each child node
    if len(node.children) != 0:
        for child in node.children:
            # Merge the child node dictionary into the children_node dictionary
            children_node = {**children_node, **get_node_dict(child)}

    temp = {}
    # Process content nodes
    if node.type_ == 'content':
        if len(node.children) == 0:
            temp[node.key] = node.value
        else:
            temp[node.value] = children_node

    elif node.type_ == 'unordered_bullet':
        if len(node.children) == 0:
            temp[node.key] = node.value
        else:
            temp[node.value] = children_node

    # Process heading nodes
    elif node.type_ == 'heading':
        if len(node.children) == 0:
            temp[node.key] = node.value
        else:
            temp[node.key + '>>' + node.value] = children_node

    # Process ordered bullet nodes
    elif node.type_ == 'ordered_bullet':
        if len(node.children) == 0:
            temp['(' + node.key + ')'] = node.value
        else:
            temp['(' + node.key + ') ' + node.value] = children_node

    return temp


def process_heading(structured_data, sentence, index, node_route):
    """
    Processes a heading and updates the structured data and node route.
    """
    head_node = Node('heading' + ''.join(str(i) for i in node_route) + str(index), sentence, 'heading')
    structured_data.append(head_node)
    latest_head_index = len(structured_data) - 1
    node_route = [latest_head_index]
    return node_route, latest_head_index


def process_ordered_bullet(structured_data, sentence, bullet, value,
                           node_route, bullet_root_indexes, latest_head_index):
    """
    Processes an ordered bullet point and updates the structured data and node route.
    """
    current_route = []
    previous_bullet = get_previous_bullet(bullet)
    if previous_bullet is not None:
        if previous_bullet in bullet_root_indexes:
            if latest_head_index > bullet_root_indexes[previous_bullet][0]:
                current_route = node_route
            else:
                current_route = bullet_root_indexes[previous_bullet]
        else:
            current_route = node_route
    else:
        current_route = node_route

    node_route = current_route
    bullet_root_indexes[bullet] = node_route.copy()
    ordered_bullet_node = Node(bullet, value, 'ordered_bullet')
    idx = add_node_child(structured_data, node_route, ordered_bullet_node)
    node_route.append(idx)
    return node_route, bullet_root_indexes


def process_unordered_bullet(structured_data, sentence, value, index, node_route):
    """
    Processes an unordered bullet point and updates the structured data and node route.
    """
    unordered_bullet_node = Node('ubullet' + ''.join(str(i) for i in node_route) + str(index), value,
                                 'unordered_bullet')
    idx = add_node_child(structured_data, node_route, unordered_bullet_node)
    node_route.append(idx)
    return node_route


def process_context(structured_data, sentence, index, node_route):
    """
    Processes a context sentence and updates the structured data and node route.
    """
    while len(node_route) > 4:
        node_route.pop()
    if get_node_type(structured_data, node_route) == 'content':
        node_route.pop()
    content_node = Node('optional-context' + ''.join(str(i) for i in node_route) + str(index), sentence, 'content')
    idx = add_node_child(structured_data, node_route, content_node)
    node_route.append(idx)
    return node_route


def is_heading(phrase):
    """
    Determines if a given phrase qualifies as a heading based on specific criteria.
    - phrase: The string phrase to evaluate.
    Returns True if the phrase meets the criteria for a heading, otherwise False.
    """

    # Check if the first character is uppercase
    if not phrase[0].isupper():
        return False

    # Check if the last character is not a symbol
    if phrase[-1] in (set(string.punctuation) - {']', ')', '—'}):
        return False

    if re.search(regex_patterns.part_division_headings, phrase):
        return True

    # Check if the number of words is 6 or less
    words = phrase.split()
    if len(words) > 6:
        return False

    if re.search(regex_patterns.ending_with_numbers, phrase):
        return False

    if re.search(regex_patterns.function_words_ending, phrase):
        return False

    return True


def get_node_value(stack, node_route):
    """
    Retrieves the value of a node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the desired node.
    Returns the value of the node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:  # Traverse the path to the desired node
        node = node.children[idx]
    return node.value  # Return the value of the final node


def get_node_children(stack, node_route):
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:  # Traverse the path to the parent node
        node = node.children[idx]  # Add the new child node
    return node.children


def get_node_dict_v2(node, index):
    children_node = {}
    # Recursively process each child node
    if len(node.children) != 0:
        for index, child in enumerate(node.children):
            # Merge the child node dictionary into the children_node dictionary
            children_node = {**children_node, **get_node_dict_v2(child, index + 1)}

    if len(children_node.keys()) == 1:
        if (not starts_with_bullet(list(children_node.keys())[0]) and
                (not children_node[list(children_node.keys())[0]] is None) and
                (not isinstance(children_node[list(children_node.keys())[0]], str))):
            node.value += '; ' + list(children_node.keys())[0]
            children_node = children_node[list(children_node.keys())[0]]

    temp = {}
    if len(node.children) == 0:
        if node.type_ >= 40:
            key, value = is_bullet(node.value)
            temp[key] = value
        elif node.type_ == 0:
            temp['heading' + str(index)] = node.value
        elif node.type_ in [1, 2, 3]:
            temp['content' + str(index)] = node.value
    else:
        children_keys = list(children_node.keys())
        if len(children_keys) == 1 and children_keys[0].startswith('content'):
            temp[node.value] = children_node[children_keys[0]]
        else:
            temp[node.value] = children_node

    return temp


def get_node_key(stack, node_route):
    """
    Retrieves the key of a node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the desired node.
    Returns the key of the node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:  # Traverse the path to the desired node
        node = node.children[idx]
    return node.key  # Return the key of the final node


def get_node_type(stack, node_route):
    """
    Retrieves the type of a node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the desired node.
    Returns the type of the node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:  # Traverse the path to the desired node
        node = node.children[idx]
    return node.type_  # Return the type of the final node


def set_node_value(stack, node_route, new_value):
    """
    Sets a new value for a node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the desired node.
    - new_value: The new value to be set for the node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:  # Traverse the path to the desired node
        node = node.children[idx]
    node.value = new_value  # Set the new value for the final node


def add_node_child(stack, node_route, new_node):
    """
    Adds a new child node to an existing node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the parent node.
    - new_node: The new child node to be added.
    Returns the index of the newly added child node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:  # Traverse the path to the parent node
        node = node.children[idx]
    node.children.append(new_node)  # Add the new child node
    return len(node.children) - 1  # Return the index of the newly added child


def get_previous_bullet(bullet):
    """
    Determines the previous bullet in a sequence based on the current bullet.
    - bullet: The current bullet string to check.
    Returns the previous bullet in the sequence, or None if it cannot be determined.
    """
    # Check for bullet points with parentheses, e.g., (5)
    match = re.match(regex_patterns.number_parenthese_bullet, bullet)
    if match:
        num = int(match.group(1))
        if num > 1:
            return f"({num - 1})"
        return None

    # Check for bullet points with parentheses and sub-numbers, e.g., 4(4)
    match = re.match(regex_patterns.number_with_number_parenthese_bullet, bullet)
    if match:
        num = int(match.group(1))
        sub_num = int(match.group(2))
        if sub_num > 1:
            return f"{num}({sub_num - 1})"
        elif num > 1:
            return f"{num - 1}"
        return None

    # Check for bullet points with a numeric suffix (e.g., 4c)
    match = re.match(regex_patterns.number_alpa_char_bullet, bullet)
    if match:
        num = int(match.group(1))
        alpha = match.group(2)
        if alpha > 'a':
            return f"{num}{chr(ord(alpha) - 1)}"
        return None

    # Check for float with numeric fractional part (e.g., 4.2)
    match = re.match(regex_patterns.float_number_bullet, bullet)
    if match:
        int_part = int(match.group(1))
        frac_part = int(match.group(2))
        if frac_part > 1:
            return f"{int_part}.{frac_part - 1}"
        return None

    # Check for float with alphabetical fractional part (e.g., 4.a)
    match = re.match(regex_patterns.float_alpha_bullet, bullet)
    if match:
        int_part = int(match.group(1))
        alpha = match.group(2)
        if alpha > 'a':
            return f"{int_part}.{chr(ord(alpha) - 1)}"
        return None

    # Check for simple numeric bullet points (e.g., 4)
    match = re.match(regex_patterns.simple_number_bullet, bullet)
    if match:
        num = int(match.group(1))
        if num > 1:
            return str(num - 1)
        return None

    # Check for simple alpha bullet points (e.g., a)
    match = re.match(regex_patterns.simple_alpha_bullet, bullet)
    if match:
        alpha = match.group(1)
        if alpha > 'a':
            return chr(ord(alpha) - 1)
        return None

    return None


def is_bullet(phrase):
    """
    Check if a given phrase matches any bullet pattern and extract the bullet and the text after the bullet.
    - phrase: The input text phrase to check.
    Returns a tuple (bullet, text_after_bullet).
    If no bullet is matched, returns (None, phrase).
    """
    for key, pattern in regex_patterns.bullet_patterns.items():
        match = re.match(pattern, phrase, re.IGNORECASE)
        if match:
            if key in ['numbered_lettered', 'numbered_numbered']:
                bullet = match.group(1) + '.' + match.group(2)  # Bullet type or symbol
                text_after_bullet = match.group(3).strip() if len(match.groups()) > 2 else match.group(2).strip()
                return bullet, text_after_bullet
            if key in ['numbered', 'lettered', 'single_letter', 'mixed_numbered', 'mixed_numbered_dot']:
                bullet = match.group(1)  # Bullet type or symbol
                text_after_bullet = match.group(2).strip()  # Text after the bullet
                return bullet, text_after_bullet
            if key in ['parentheses_number']:
                bullet = '(' + match.group(1) + ')'  # Bullet type or symbol
                text_after_bullet = match.group(2).strip()  # Text after the bullet
                return bullet, text_after_bullet
            elif key in ['mixed_numbered_parentheses']:
                bullet = match.group(1) + '(' + match.group(2) + ')'  # Bullet type or symbol
                text_after_bullet = match.group(3).strip() if len(match.groups()) > 2 else match.group(
                    2).strip()  # Text after the bullet
                return bullet, text_after_bullet
            elif key in ['bullet_symbol', '*', '-', '•']:
                bullet = '*'  # Bullet type or symbol
                text_after_bullet = match.group(2).strip()  # Text after the bullet
                return bullet, text_after_bullet

    # Default return (None, '') if no bullet is matched
    return None, phrase


def extract_pdf_raw_text(pdf_path):
    """
    Extracts raw text from a PDF file.
    - pdf_path: The path to the PDF file.
    Returns a tuple containing:
    - content_phrases_by_pages: A list where each element is a list of text phrases (lines) from a page.
    - extracted_text: A single string containing all the extracted text.
    """
    logging.info(logging_messages.extracting_file.format(pdf_path))
    content_phrases_by_pages = []
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text
                    content_phrases_by_pages.append(text.split('\n'))
    except Exception as e:
        logging.error(logging_messages.error_extracting.format(e))
        return None, None

    return content_phrases_by_pages, extracted_text


def extract_pdf_metadata(pdf_path):
    logging.info(logging_messages.extracting_file.format(pdf_path))
    # Open the PDF file
    doc = fitz.open(pdf_path)

    pages_list = []
    # Iterate through the pages
    try:
        for page in doc:
            spans_list = []
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if 'lines' in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            spans_list.append(span)

            pages_list.append(spans_list)
    except Exception as e:
        logging.error(logging_messages.error_extracting.format(e))
        return None

    return pages_list


def get_level_path(level, current_level_path):
    level_path = []
    for lvl in current_level_path:
        level_path.append(lvl)
        if lvl == level:
            break

    return level_path


def classify_page_text_by_levels(pages_list):
    logging.info(logging_messages.classify_levels)
    type_level = {}
    level = 1

    classified_pages = []

    for page in pages_list:
        classified_spans = []
        for span in page:
            key = strings.classify_level_string.format(span['flags'], span['color'], span['font'], span['size'])
            if key not in type_level:
                type_level[key] = level
                span['level'] = level
                level += 1

            span['level'] = type_level[key]

            if len(classified_spans) == 0:
                classified_spans.append(span)
            elif span['text'].strip() == '':
                if classified_spans[-1]['text'].strip() != '':
                    classified_spans.append(span)
            elif classified_spans[-1]['text'].strip() == '':
                classified_spans.append(span)
            elif span['level'] == classified_spans[-1]['level']:
                classified_spans[-1]['text'] += span['text']
            else:
                classified_spans.append(span)

        classified_pages.append(classified_spans)

    return classified_pages


def classify_page_text_by_types(formatted_pages):
    logging.info(logging_messages.classify_sentences)
    type_classified_pages = []

    available_bullet_index = 40
    found_bullets = {}
    for span in formatted_pages:
        bullet, value = is_bullet(span['text'])
        if bullet is not None:
            previous_bullet = get_previous_bullet(bullet)
            if previous_bullet is not None:
                if previous_bullet in found_bullets:
                    span['type_index'] = found_bullets[previous_bullet]
                    found_bullets[bullet] = found_bullets[previous_bullet]
                else:
                    span['type_index'] = available_bullet_index
                    found_bullets[bullet] = available_bullet_index
                    available_bullet_index += 1
            else:
                span['type_index'] = available_bullet_index
                found_bullets[bullet] = available_bullet_index
                available_bullet_index += 1

        elif span['text'][0].isupper():
            if ends_with_special_v2(span['text'].strip()):
                span['type_index'] = 3
            else:
                span['type_index'] = 0
        else:
            if ends_with_special_v2(span['text'].strip()):
                span['type_index'] = 2
            else:
                span['type_index'] = 1

        type_classified_pages.append(span)

    return type_classified_pages


def clean_non_alpha_char_list(elements):
    """Clean the elements by removing non-alphabetic characters and extract the cleaned list of elements."""
    return [re.sub(regex_patterns.alpha_characters_only, '', element) for element in elements]


def clean_non_alpha_char_single(element):
    """Clean the elements by removing non-alphabetic characters and extract the cleaned element."""
    return re.sub(regex_patterns.alpha_characters_only_v2, '', element, flags=re.IGNORECASE)


def get_candidates(page, start_idx, end_idx):
    """Extract and clean candidates from a specific section of each page."""
    return clean_non_alpha_char_list(page[start_idx:end_idx]) if len(page) >= constants.header_estimated_size else []


def get_candidates_v2(page, end_count):
    """Extract and clean candidates from a specific section of each page."""
    if len(page) > end_count:
        return [strings.header_footer_level_setup_string
                .format(str(span['level']), clean_non_alpha_char_single(span['text'].strip()))
                for span in page if span['text'].strip()][:end_count]
    else:
        return []


def filter_candidates(counter, threshold):
    """Filter the candidates based on the occurrence threshold."""
    return [(candidate, count) for candidate, count in counter.items() if count > threshold]


def extract_candidates(pages, start_idx, end_idx, counter):
    """Extract candidates from each page and update the counter."""
    for page in pages:
        candidates = get_candidates(page, start_idx, end_idx)
        counter.update(candidates)


def extract_candidates_v2(pages, end_count, counter):
    """Extract candidates from each page and update the counter."""
    for page in pages:
        candidates = get_candidates_v2(page, end_count)
        counter.update(candidates)


def extract_headers_and_footers(pages):
    """Extract and filter headers and footers from the given pages."""
    logging.info(logging_messages.identify_header_footer)
    header_counter = Counter()
    footer_counter = Counter()

    # Extract header and footer candidates
    extract_candidates(pages, 0, constants.header_estimated_size, header_counter)
    extract_candidates(pages, -constants.footer_estimated_size, None, footer_counter)

    # Filter candidates based on occurrence threshold
    filtered_headers = filter_candidates(header_counter, constants.header_footer_occurrence_accept_threshold)
    filtered_footers = filter_candidates(footer_counter, constants.header_footer_occurrence_accept_threshold)

    return filtered_headers, filtered_footers


def extract_headers_and_footers_v2(pages):
    logging.info(logging_messages.identify_header_footer)

    header_footer_counter = Counter()

    extract_candidates_v2(pages, constants.header_footer_estimated_size_v2, header_footer_counter)

    filtered_headers_footers = filter_candidates(header_footer_counter,
                                                 constants.header_footer_occurrence_accept_threshold_v2)

    header_footer_levels = []

    for x in filtered_headers_footers:
        level = int(x[0].split('->')[0])
        if level not in header_footer_levels:
            header_footer_levels.append(level)

    return header_footer_levels


def is_header_or_footer(line, headers, footers):
    """Check if the line is one of the identified header or footer."""
    cleaned_line = re.sub(regex_patterns.alpha_characters_only, '', line)
    if any(cleaned_line == header[0] for header in headers):
        return constants.header_tag
    if any(cleaned_line == footer[0] for footer in footers):
        return constants.footer_tag
    return None


def remove_lines(page_section, headers, footers, section_type):
    """Remove identified header or footer lines from a section of the page."""
    return set(line for line in page_section if is_header_or_footer(line, headers, footers) == section_type)


def remove_header_footer(pages, headers, footers):
    """Remove identified headers and footers from the pages."""
    logging.info(logging_messages.remove_header_footer)
    content_pages = []

    for page in pages:
        if len(page) <= constants.header_estimated_size:
            content_pages.append(page)
            continue

        # Identify header and footer lines to remove
        header_to_remove = remove_lines(page[:constants.header_estimated_size], headers, footers, constants.header_tag)
        footer_to_remove = remove_lines(page[-constants.footer_estimated_size:], headers, footers, constants.footer_tag)

        # Clean page by removing identified header/footer lines
        to_remove = header_to_remove.union(footer_to_remove)
        content_page = [line for line in page if line not in to_remove]
        content_pages.append(content_page)

    return content_pages


def remove_header_footer_v2(pages, header_footer_levels):
    logging.info(logging_messages.remove_header_footer)
    cleaned_data = []
    for index, page in enumerate(pages):
        if len(page) > 6:
            count = 0
        else:
            count = 10
        for span in page:
            if span['level'] in header_footer_levels and span['text'].strip() != '' and count < 10:
                count += 1
            else:
                cleaned_data.append(span)
                continue

    return cleaned_data


def clean_text_by_formats_v2(cleaned_data):
    logging.info(logging_messages.group_sentences)
    formatted_data = []
    temp_span = None
    not_ended = False
    for span in cleaned_data:
        if temp_span is None:
            if span['text'].strip() == '':
                continue
            if ends_with_special_v2(span['text'].strip()):
                formatted_data.append(span)
            else:
                if not_ended_with_special(span['text'].strip()):
                    not_ended = True
                temp_span = span
            continue
        if (span['text'].strip() == '' or (span['text'][0].isupper() and not not_ended)
                or starts_with_bullet(span['text'])):
            formatted_data.append(temp_span)
            if span['text'].strip() == '':
                temp_span = None
                not_ended = False
                continue
            else:
                temp_span = span
        else:
            if not_ended_with_special(span['text'].strip()):
                not_ended = True
            else:
                not_ended = False
            temp_span['text'] += span['text']

        if ends_with_special_v2(temp_span['text'].strip()):
            formatted_data.append(temp_span)
            temp_span = None
            not_ended = False

    return formatted_data


def clean_text_by_types_v2(type_formatted_pages):
    clean_data = {
        0: type_formatted_pages
    }
    version = 1
    need_merges = True
    while need_merges:
        clean_data[version] = []
        need_merges = False
        for span in clean_data[version - 1]:
            if len(clean_data[version]) != 0:
                if span['type_index'] == 1:
                    if clean_data[version][-1]['type_index'] == 0:
                        clean_data[version][-1]['text'] += f" {span['text']}"
                        continue
                    elif clean_data[version][-1]['type_index'] == 1:
                        clean_data[version][-1]['text'] += f" {span['text']}"
                        continue
                elif span['type_index'] == 2:
                    if clean_data[version][-1]['type_index'] == 0:
                        clean_data[version][-1]['text'] += f" {span['text']}"
                        clean_data[version][-1]['type_index'] = 3
                        need_merges = True
                        continue
                    elif clean_data[version][-1]['type_index'] == 1:
                        clean_data[version][-1]['text'] += f" {span['text']}"
                        clean_data[version][-1]['type_index'] = 2
                        need_merges = True
                        continue
            clean_data[version].append(span)
        version += 1

    extractable_data = []

    for span in clean_data[version - 1]:
        if len(extractable_data) != 0:
            if span['type_index'] == 3:
                if extractable_data[-1]['type_index'] == 0:
                    extractable_data[-1]['text'] += f" {span['text']}"
                    extractable_data[-1]['type_index'] = 3
                    continue
                elif extractable_data[-1]['type_index'] == 1:
                    extractable_data[-1]['text'] += f" {span['text']}"
                    extractable_data[-1]['type_index'] = 2
                    continue
        extractable_data.append(span)

    return extractable_data


def download_nltk_resources():
    """Download necessary NLTK resources."""
    for package in constants.nltk_resource_packages:
        try:
            nltk.data.find(strings.nlt_source_path.format(package))
            logging.info(logging_messages.nltk_resource_downloaded.format(package))
        except LookupError:
            logging.info(logging_messages.nltk_resource_downloading.format(package))
            nltk.download(package)
            logging.info(logging_messages.status_success)


def starts_with_bullet(phrase):
    """Check if the phrase starts with a bullet point."""
    return bool(re.match(regex_patterns.start_with_bullet_pattern, phrase, re.IGNORECASE))


def ends_with_special(phrase):
    """Check if the phrase ends with special characters."""
    return bool(re.search(regex_patterns.ends_with_special, phrase))


def ends_with_special_v2(phrase):
    return phrase[-1] in [';', '.', '—']


def not_ended_with_special(phrase):
    return phrase[-1] in [':', '-', ',']


def clean(text):
    """
    Clean the input text by applying several transformations:
    - Remove leading and trailing whitespace
    - Remove newline characters before colons
    - Remove extra spaces between words
    - Remove spaces before full stops
    - Remove spaces between newlines
    - Replace multiple consecutive newlines with a single newline
    """
    text = text.strip()
    text = re.sub(re.compile(regex_patterns.newline_before_colon), ' ', text)
    text = re.sub(regex_patterns.extra_space_btw_words, ' ', text)
    text = text.replace(' .', '.')
    text = text.replace('\n ', '\n')
    text = re.sub(regex_patterns.multiple_consecutive_newlines, '\n\n', text)

    return text


def get_file_name(path):
    """return the file name of the given path"""
    return path.split('/')[-1]


def build_dataset_v0(text, file_name, embedder, extract_version):
    """
    Convert chunks of text into a dataset format compatible with the 'datasets' library.
    - chunks: List of text chunks to be converted.
    - file_name: The name of the source file.
    """
    chunk_dictlist = create_chunk_dictlist_v0(text, file_name, embedder, extract_version)
    return Dataset.from_list(chunk_dictlist)


def build_dataset_v1(json_dict, file_name, embedder, extract_version):
    """
    Convert chunks of text into a dataset format compatible with the 'datasets' library.
    - chunks: List of text chunks to be converted.
    - file_name: The name of the source file.
    """
    chunk_dictlist = create_chunk_dictlist_v1(json_dict, file_name, embedder, extract_version)
    return Dataset.from_list(chunk_dictlist)


def build_dataset_v2(json_dict, file_name, embedder, extract_version):
    """
    Convert chunks of text into a dataset format compatible with the 'datasets' library.
    - chunks: List of text chunks to be converted.
    - file_name: The name of the source file.
    """
    chunk_dictlist = create_chunk_dictlist_v2(json_dict, file_name, embedder, extract_version)
    return Dataset.from_list(chunk_dictlist)


def chunk_text_by_token_limit_v0(text, embedder):
    current_sentence = ''
    chunk_texts = []

    for sentence in text:
        temp_sentence = current_sentence + '\n' + sentence
        token_count = embedder.count_tokens(temp_sentence)
        if token_count <= constants.chunk_token_limit:
            current_sentence = temp_sentence
        else:
            chunk_texts.append(current_sentence)
            current_sentence = sentence

    chunk_texts.append(current_sentence)

    return chunk_texts


def create_chunk_dictlist_v0(text, file_name, embedder, extract_version):

    chunk_texts = chunk_text_by_token_limit_v0(text, embedder)

    chunk_dictlist = [{'id': file_name.split('.')[0] + '-' + str(i + 1),
                       'metadata': {
                           'content': chunk_content,
                           'source': file_name
                       }}
                      for i, chunk_content in enumerate(chunk_texts)]

    save_preprocessed_data('chunks/by_token_limit',
                           json.dumps(chunk_dictlist, indent=2), '/' + file_name,
                           extract_version, 'chunks', 'json')
    return chunk_dictlist


def create_chunk_dictlist_v1(json_dict, file_name, embedder, extract_version):
    chunks = []
    for head_lvl1 in json_dict.keys():
        process_chunks_to_lowest_node(json_dict[head_lvl1], [head_lvl1], chunks, embedder)

    chunk_texts = merge_chunks_to_token_limit(chunks, embedder)

    chunk_dictlist = [{'id': file_name.split('.')[0] + '-' + str(i + 1),
                       'metadata': {
                           'content': str(chunk),
                           'source': file_name
                       }}
                      for i, chunk in enumerate(chunk_texts)]

    save_preprocessed_data('chunks/by_token_limit',
                           json.dumps(chunk_dictlist, indent=2), '/' + file_name,
                           extract_version, 'chunks', 'json')

    return chunk_dictlist


def process_chunks_to_lowest_node(list_, heading, chunks_, embedder):
    data = {}
    update_nested_dict(data, heading, list_)
    token_count = embedder.count_tokens(json.dumps(data))
    if token_count > constants.chunk_token_limit:
        if isinstance(list_, dict):
            for attach_head in list_.keys():
                new_heading = heading.copy()
                new_heading.append(attach_head)
                chunks_ = process_chunks_to_lowest_node(list_[attach_head], new_heading, chunks_, embedder)
            return chunks_
        else:
            chunks_.append(data)
            return chunks_
    else:
        chunks_.append(data)
        return chunks_


def merge_dicts(dict1, dict2):
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            else:
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1


def segment_text_by_sentences(text, num_sections):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    # Calculate the approximate number of sentences per section
    sentences_per_section = len(sentences) // num_sections

    segments = []
    start = 0

    # Loop to create sections
    for i in range(num_sections - 1):
        # Calculate the end index for this section
        end = start + sentences_per_section
        # Join the sentences to form a segment
        segment = ' '.join(sentences[start:end])
        segments.append(segment)
        start = end

    # Handle the last segment, which may contain the remaining sentences
    segments.append(' '.join(sentences[start:]))

    return segments


def merge_chunks_to_token_limit(chunks, embedder):
    merged_chunks = []
    temp_chunk = {}
    for chunk in chunks:
        chunk_token_count = embedder.count_tokens(json.dumps(chunk))
        if chunk_token_count < constants.chunk_token_limit:
            # Create a deep copy of temp
            temp_copy = copy.deepcopy(temp_chunk)
            test_chunk = merge_dicts(temp_copy, chunk)
            test_token_count = embedder.count_tokens(json.dumps(test_chunk))
            if test_token_count < constants.chunk_token_limit:
                temp_chunk = test_chunk
            else:
                merged_chunks.append(temp_chunk)
                temp_chunk = chunk
        else:
            merged_chunks.append(temp_chunk)
            temp_chunk = {}

            # handle big chunk text strings
            segments = math.ceil(chunk_token_count / constants.chunk_token_limit)
            headings = []
            while isinstance(chunk, dict):
                head = list(chunk.keys())[0]
                headings.append(head)
                chunk = chunk[head]

            segmented_contents = segment_text_by_sentences(chunk, segments)
            for content in segmented_contents:
                sub_chunk = {}
                update_nested_dict(sub_chunk, headings, content)
                merged_chunks.append(sub_chunk)

    if temp_chunk != {}:
        merged_chunks.append(temp_chunk)

    return merged_chunks


def title_chunks(chunks):
    titled_chunks = []
    for chunk in chunks:
        title = ''
        while isinstance(chunk, dict) and len(chunk.keys()) == 1:
            head = list(chunk.keys())[0]
            if title == '':
                title = head
            else:
                title += ' >> ' + head
            chunk = chunk[head]

        titled_chunks.append({'title': title, 'content': chunk})

    return titled_chunks


def create_chunk_dictlist_v2(json_dict, file_name, embedder, extract_version):
    chunks = []
    for head_lvl1 in json_dict['structured_data'].keys():
        process_chunks_to_lowest_node(json_dict['structured_data'][head_lvl1], [head_lvl1], chunks, embedder)

    limit_merged_chunks = merge_chunks_to_token_limit(chunks, embedder)
    chunk_title_content = title_chunks(limit_merged_chunks)

    chunk_dictlist = [{'id': file_name.split('.')[0] + '-' + str(i + 1),
                       'metadata': {
                           'title': chunk['title'],
                           'content': str(chunk['content']),
                           'source': file_name
                       }}
                      for i, chunk in enumerate(chunk_title_content)]

    save_preprocessed_data('chunks/by_token_limit',
                           json.dumps(chunk_dictlist, indent=2), '/' + file_name,
                           extract_version, 'chunks', 'json')

    return chunk_dictlist


def process_chunks(list_, heading, level, chunks_, embedder):
    content = {heading: list_}
    if level <= constants.chunk_window_size:
        if isinstance(list_, dict):
            for attach_head in list_.keys():
                new_heading = heading + ' >> ' + attach_head
                chunks_ = process_chunks(list_[attach_head], new_heading, level + 1, chunks_, embedder)
            return chunks_
        else:
            chunks_.append(content)
            return chunks_
    else:
        chunks_.append(content)
        return chunks_


def unstruct_process_docs(docs):
    return '\n----\n'.join(docs)


def merge_chunks_v1(docs):
    merged_docs = {}
    for doc in docs:
        doc_dict = convert_string_to_dict(doc)
        for key in doc_dict.keys():
            merged_docs[key] = doc_dict[key]

    return merged_docs


def v1_json_process_docs(docs):
    return json.dumps(merge_chunks_v1(docs))


def v1_html_process_docs(docs):
    return convert_to_html(merge_chunks_v1(docs))


def update_nested_dict(d, keys, value):
    # The first key to start processing
    key = keys[0]

    # If there's only one key left in the list, assign the value
    if len(keys) == 1:
        d[key] = value
    else:
        # If the key does not exist or it's not a dictionary, create a dictionary for it
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}

        # Recursively call the function with the rest of the keys
        update_nested_dict(d[key], keys[1:], value)


def update_nested_dict_v2(d, keys, value):
    key = keys[0]

    # If there's only one key left in the list, merge the value if it's a dict or directly assign
    if len(keys) == 1:
        if isinstance(value, dict) and isinstance(d.get(key), dict):
            merge_dicts(d[key], value)
        else:
            d[key] = value
    else:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        update_nested_dict_v2(d[key], keys[1:], value)


def convert_string_to_dict(context):
    try:
        return ast.literal_eval(context)
    except (ValueError, SyntaxError):
        print(f"{context}: The string is not a valid dictionary format.")
        return context


def merge_chunks_v2(docs):
    structured_docs = {}
    for doc in docs:
        heading_str = doc['title']
        context = doc['content']
        context = convert_string_to_dict(context)

        headings = heading_str.split(' >> ')
        if heading_str == '':
            merge_dicts(structured_docs, context)
        else:
            update_nested_dict_v2(structured_docs, headings, context)

    return structured_docs


def v2_json_process_docs(docs):
    return json.dumps(merge_chunks_v2(docs))


def convert_sub_html_content(data, level):
    html_string = ''
    if isinstance(data, dict):
        keys = list(data.keys())
        for key in keys:
            html_string += f'<span><h{level}>{key}</h{level}>'
            html_string += convert_sub_html_content(data[key], level + 1)
            html_string += '</span>'
    else:
        html_string += f'<p>{data}</p>'

    return html_string


def convert_to_html(structured_docs):
    head_keys = list(structured_docs.keys())
    html_string = ''
    for head in head_keys:
        html_string += f'<div><h1>{head}</h1>'
        html_string += convert_sub_html_content(structured_docs[head], 2)
        html_string += '</div>'

    return html_string


def v2_html_process_docs(docs):
    structured_docs = merge_chunks_v2(docs)
    return convert_to_html(structured_docs)


def convert_to_custom1(structured_docs):
    head_keys = list(structured_docs.keys())
    custom_string = ''
    for head in head_keys:
        custom_string += f'<div><h1>{head}</h1>'
        custom_string += f'<p>{json.dumps(structured_docs[head])}</p>'
        custom_string += '</div>'

    return custom_string


def v2_custom1_process_docs(docs):
    structured_docs = merge_chunks_v2(docs)
    return convert_to_custom1(structured_docs)


def convert_to_custom2(structured_docs):
    head_keys = list(structured_docs.keys())
    custom_v2 = {}
    for head in head_keys:
        custom_v2[head] = convert_sub_html_content(structured_docs[head], 1)

    return json.dumps(custom_v2)


def v2_custom2_process_docs(docs):
    structured_docs = merge_chunks_v2(docs)
    return convert_to_custom2(structured_docs)


def convert_to_custom3(structured_docs, headings):
    head_keys = list(structured_docs.keys())
    toml_string = ''
    for head in head_keys:
        new_headings = headings.copy()
        new_headings.append(head)
        if isinstance(structured_docs[head], dict):
            toml_string += convert_to_toml(structured_docs[head], new_headings)
        else:
            toml_string += f'["{'.'.join(new_headings)}"]\n'
            toml_string += f'title = "{head}"\n'
            toml_string += f'description = "{convert_sub_html_content(structured_docs[head], 1)}"\n\n'

    return toml_string


def v2_custom3_process_docs(docs):
    structured_docs = merge_chunks_v2(docs)
    return convert_to_custom3(structured_docs, ['Context'])


def convert_sub_markdown_content(data, level):
    md_string = ''
    if isinstance(data, dict):
        keys = list(data.keys())
        for key in keys:
            md_head_tag = '#'
            md_string += f'{md_head_tag * level} {key}\n'
            md_string += convert_sub_markdown_content(data[key], level + 1)
    else:
        md_string += f'{data}\n\n'

    return md_string


def convert_to_markdown(structured_docs):
    head_keys = list(structured_docs.keys())
    md_string = ''
    for head in head_keys:
        md_string += f'# {head}\n'
        md_string += convert_sub_markdown_content(structured_docs[head], 2)

    return md_string


def v2_markdown_process_docs(docs):
    structured_docs = merge_chunks_v2(docs)
    return convert_to_markdown(structured_docs)


def convert_to_toml(structured_docs, headings):
    head_keys = list(structured_docs.keys())
    toml_string = ''
    for head in head_keys:
        new_headings = headings.copy()
        new_headings.append(head)
        if isinstance(structured_docs[head], dict):
            toml_string += convert_to_toml(structured_docs[head], new_headings)
        else:
            toml_string += f'["{'.'.join(new_headings)}"]\n'
            toml_string += f'title = "{head}"\n'
            toml_string += f'description = "{structured_docs[head]}"\n\n'

    return toml_string


def v2_toml_process_docs(docs):
    structured_docs = merge_chunks_v2(docs)
    return convert_to_toml(structured_docs, ['Context'])


def group_sentences(phrases):
    """
    Group phrases into sentences, handling cases like bullets and special ending characters.
    - phrases: List of phrases to be grouped.
    """
    logging.info(logging_messages.group_sentences)
    sentences = []
    temp = None

    for phrase in map(str.strip, phrases):
        if not phrase:
            continue

        if temp:
            if (starts_with_bullet(phrase)
                    or (phrase[0].isupper() and
                        not ends_with_special(phrase))):
                sentences.append(temp)
                temp = phrase
            else:
                temp += ' ' + phrase

        else:
            temp = phrase

        if ends_with_special(temp):
            sentences.append(temp)
            temp = None

    if temp:
        sentences.append(temp)

    return sentences


def create_results_dataframe(result_dict, metric):
    # Initialize the data dictionary
    data = {}

    # Collect all format types dynamically
    all_format_types = set()
    for formats in result_dict.values():
        all_format_types.update(formats.keys())

    # Initialize empty lists in the data dictionary for each format type
    for format_type in all_format_types:
        data[format_type] = []

    # Populate the data dictionary
    for format_type in all_format_types:
        for model, formats in result_dict.items():
            if format_type in formats:
                # Calculate the average score instead of the max score
                avg_score = sum(item[metric] for item in formats[format_type]) / len(formats[format_type])
            else:
                avg_score = None  # Use None if the format type is missing for a model
            data[format_type].append(avg_score)

    # Create a DataFrame and transpose it to have models on the x-axis and formats on the y-axis
    df = pd.DataFrame(data, index=result_dict.keys())
    return df.T


def visualize_rouge_results(rouge1_df, rougeL_df, format_lists, question_id):
    # Ensure the order of formats
    formats_order = [item['id'] for item in format_lists]
    rouge1_df = rouge1_df.loc[formats_order]
    rougeL_df = rougeL_df.loc[formats_order]

    # List of models
    models = rouge1_df.columns

    # Plotting the line chart
    plt.figure(figsize=(10, 6))

    # Plot ROUGE-1 scores (solid lines)
    for model in models:
        plt.plot(formats_order, rouge1_df[model], label=f'{model} (ROUGE-1)', linestyle='-', marker='o')

    # Plot ROUGE-L scores (dashed lines)
    for model in models:
        plt.plot(formats_order, rougeL_df[model], label=f'{model} (ROUGE-L)', linestyle='--', marker='o')

    # Adding labels and title
    plt.xlabel('Formats')
    plt.ylabel('ROUGE Scores')
    plt.title('ROUGE Scores by Model and Format')
    plt.legend(loc='best')  # Place legend in the best location
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Adjust y-axis to the range of the data
    plt.ylim(
        min(rouge1_df.min().min(), rougeL_df.min().min()) - 0.05,
        max(rouge1_df.max().max(), rougeL_df.max().max()) + 0.05
    )
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'results/temp/{question_id}_rouge_scores_plot.png')  # Save plot as PNG file

    plt.show()
