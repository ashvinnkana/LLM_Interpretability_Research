import re
import nltk
import logging
import pdfplumber
import string
from rouge_score import rouge_scorer

from datasets import Dataset
from nltk.tokenize import sent_tokenize
from collections import Counter

from utils import strings, constants, regex_patterns, logging_messages

from models.node import Node


def save_preprocessed_data(type_, data, unstructured_file_path, extract_version, struct_version, file_extension):
    """
    Save preprocessed data to a structured file path.

    Parameters:
    type_ (str): The type of data being processed.
    data (str): The preprocessed data to be saved.
    unstructured_file_path (str): The file path of the unstructured data.
    extract_version (str): The version of the extraction process.
    struct_version (str): The version of the structure format.
    file_extension (str): The file extension for the structured file.
    """
    file_name = (unstructured_file_path.split('/')[-1]).split('.')[0]
    structured_file_path = strings.structured_data_path.format(type_, file_name, extract_version, struct_version, file_extension)
    with open(structured_file_path, 'w') as file:
        file.write(data)


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


def extract_scores(score_dict, response):
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
    dict_['response'] = response
    return dict_


def get_rouge_scores(ref_answer, response):
    """
    Calculate the ROUGE scores between a reference answer and a generated response.

    Args:
        ref_answer (str): The reference text against which the response will be evaluated.
        response (str): The generated response text that needs to be scored.

    Returns:
        dict: A dictionary containing the F-measure scores for the specified ROUGE metrics ('rouge1', 'rougeL').
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return extract_scores(scorer.score(ref_answer, response), response)


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
    unordered_bullet_node = Node('ubullet' + ''.join(str(i) for i in node_route) + str(index), value, 'unordered_bullet')
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
    for idx in node_route[1:]:   # Traverse the path to the desired node
        node = node.children[idx]
    return node.value            # Return the value of the final node


def get_node_key(stack, node_route):
    """
    Retrieves the key of a node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the desired node.
    Returns the key of the node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:   # Traverse the path to the desired node
        node = node.children[idx]
    return node.key              # Return the key of the final node


def get_node_type(stack, node_route):
    """
    Retrieves the type of a node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the desired node.
    Returns the type of the node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:   # Traverse the path to the desired node
        node = node.children[idx]
    return node.type_            # Return the type of the final node


def set_node_value(stack, node_route, new_value):
    """
    Sets a new value for a node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the desired node.
    - new_value: The new value to be set for the node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:   # Traverse the path to the desired node
        node = node.children[idx]
    node.value = new_value       # Set the new value for the final node


def add_node_child(stack, node_route, new_node):
    """
    Adds a new child node to an existing node in a nested structure.
    - stack: A list or stack containing the root nodes.
    - node_route: A list of indices representing the path to the parent node.
    - new_node: The new child node to be added.
    Returns the index of the newly added child node.
    """
    node = stack[node_route[0]]  # Start from the root node
    for idx in node_route[1:]:   # Traverse the path to the parent node
        node = node.children[idx]
    node.children.append(new_node)  # Add the new child node
    return len(node.children) - 1   # Return the index of the newly added child


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
                bullet = match.group(1) + '.' + match.group(2)   # Bullet type or symbol
                text_after_bullet = match.group(3).strip() if len(match.groups()) > 2 else match.group(2).strip()
                return bullet, text_after_bullet
            if key in ['numbered', 'lettered', 'single_letter', 'mixed_numbered', 'mixed_numbered_dot']:
                bullet = match.group(1)  # Bullet type or symbol
                text_after_bullet = match.group(2).strip()  # Text after the bullet
                return bullet, text_after_bullet
            if key in ['parentheses_number']:
                bullet = '('+match.group(1)+')'  # Bullet type or symbol
                text_after_bullet = match.group(2).strip()  # Text after the bullet
                return bullet, text_after_bullet
            elif key in ['mixed_numbered_parentheses']:
                bullet = match.group(1) + '(' + match.group(2) + ')'   # Bullet type or symbol
                text_after_bullet = match.group(3).strip() if len(match.groups()) > 2 else match.group(2).strip()  # Text after the bullet
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

    return content_phrases_by_pages, extracted_text


def clean_non_alpha_char(elements):
    """Clean the elements by removing non-alphabetic characters and extract the cleaned elements."""
    return [re.sub(regex_patterns.alpha_characters_only, '', element) for element in elements]


def get_candidates(page, start_idx, end_idx):
    """Extract and clean candidates from a specific section of each page."""
    return clean_non_alpha_char(page[start_idx:end_idx]) if len(page) >= constants.header_estimated_size else []


def filter_candidates(counter, threshold):
    """Filter the candidates based on the occurrence threshold."""
    return [(candidate, count) for candidate, count in counter.items() if count > threshold]


def extract_candidates(pages, start_idx, end_idx, counter):
    """Extract candidates from each page and update the counter."""
    for page in pages:
        candidates = get_candidates(page, start_idx, end_idx)
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


def chunk_by_word_limit(text, max_words=500):
    """
    Split the input text into chunks, each containing a maximum number of words.
    - text: The input text to be chunked.
    - max_words: The maximum number of words allowed in each chunk (default is 500).
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def get_file_name(path):
    """return the file name of the given path"""
    return path.split('/')[-1]


def build_dataset(chunks, file_name, is_dict):
    """
    Convert chunks of text into a dataset format compatible with the 'datasets' library.
    - chunks: List of text chunks to be converted.
    - file_name: The name of the source file.
    """
    chunk_dictlist = create_chunk_dictlist(chunks, file_name, is_dict)
    return Dataset.from_list(chunk_dictlist)


def create_chunk_dictlist(chunks, file_name, is_dict=False):
    """
    Create a list of dictionaries for each chunk, suitable for creating a dataset.
    - chunks: List or dict of text chunks to be converted.
    - file_name: The name of the source file.
    - is_dict: Boolean indicating whether chunks are a dictionary.
    """
    chunk_dictlist = []
    if is_dict:
        keys = [key for key in chunks.keys() if isinstance(chunks[key], dict)]
        chunk_dictlist = [{'id': str(i + 1),
                           'metadata': {
                               'content': f'{key} : {str(chunks[key])}',
                               'source': file_name
                           }}
                          for i, key in enumerate(keys)]
    else:
        chunk_dictlist = [{'id': str(i + 1),
                           'metadata': {
                               'content': chunk_content,
                               'source': file_name
                           }}
                          for i, chunk_content in enumerate(chunks)]
    return chunk_dictlist


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
