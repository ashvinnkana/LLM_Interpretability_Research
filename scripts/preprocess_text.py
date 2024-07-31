import re
import nltk
from datasets import Dataset
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


def clean(text):
    # Remove extra line breaks within sentences
    text = re.sub(r'\n(?=[a-z])', ' ', text)

    # Remove leading and trailing whitespace
    text = text.strip()

    # remove newline before colon
    text = re.sub(re.compile(r'\n(?=:)'), ' ', text)

    # fix numbering patterns
    text = re.sub(re.compile(r'(\d)\.\s*\n'), r'\1. ', text)

    # Remove extra spaces between words
    text = re.sub(r'(?<!\n) +', ' ', text)

    # remove spaces before fullstops
    text = text.replace(' .', '.')

    # remove spaces between newlines
    text = text.replace('\n ', '\n')

    # Replace multiple consecutive newlines with a single newline
    text = re.sub(r'\n\n+', '\n\n', text)

    return text


def chunk_by_word_limit(text, max_words=500):
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


def convert_to_dataset(chunks):
    chunk_dictlist = [{'id': str(i + 1), 'metadata': {'content': chunk_content}} for i, chunk_content in enumerate(chunks)]
    return Dataset.from_list(chunk_dictlist)
