from scripts import extract_file, clean_text, llm
from utils import constants


def main():
    # Extract PDF Contents
    data = extract_file.extract_pdf('./data/unstructured_data/wa_legislation_files/Limitation_Act_2005.pdf')
    data = clean_text.basic(data)

    # print content of the file
    #print(data)

    # TODO: Ask a question passing this extracted text to LLM

    legal_llm = llm.LLM(constants.aus_legal_llm)
    tokens = legal_llm.tokenize('Section 51 of the Constitution provides')
    print(legal_llm.generate_response(tokens))


if __name__ == '__main__':
    main()
