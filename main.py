from scripts import extract_file, clean_text


def main():
    # Extract PDF Contents
    data = extract_file.extract_pdf('./data/unstructured_data/wa_legislation_files/Limitation_Act_2005.pdf')
    data = clean_text.basic(data)

    # print content of the file
    print(data)


if __name__ == '__main__':
    main()
