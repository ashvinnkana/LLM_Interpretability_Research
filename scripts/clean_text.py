import re


def basic(text):
    # Remove extra line breaks within sentences
    text = re.sub(r'\n(?=[a-z])', ' ', text)

    # Remove leading and trailing whitespace
    text = text.strip()

    # Remove extra spaces
    # text = re.sub(r'\s+', ' ', text).strip()

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

    # text = text.replace('\n', '\\n')

    return text
