import unittest

from utils import constants
from utils.functions import get_node_children, extract_pdf_metadata, classify_page_text_by_levels, \
    extract_headers_and_footers_v2, remove_header_footer_v2, clean_text_by_formats_v2, classify_page_text_by_types, \
    clean_text_by_types_v2, extract_data_v2


def run_tests():
    class TestExtraction(unittest.TestCase):
        def test_with_extracted_headings(self):
            input_pdf = './test_data/Limitation_Act_2005.pdf'
            input_node_route = [0, 1, 0, 0]
            output_expected = [
                'Part 1  Preliminary',
                'Part 2  Limitation periods',
                'Part 3  Extension or shortening of limitation periods',
                'Part 4  Accrual of particular causes of action',
                'Part 5  Effect of expiration of limitation period',
                'Part 6  Miscellaneous',
                'Part 7  Transitional provisions  [Heading inserted No. 3 of 2018 s. 12.]']
            output_obtained = []

            pages = extract_pdf_metadata(input_pdf)

            lvl_classified_pages = classify_page_text_by_levels(pages)

            header_footer_levels = extract_headers_and_footers_v2(lvl_classified_pages)
            content_pages = remove_header_footer_v2(lvl_classified_pages, header_footer_levels)

            cleaned_pages = clean_text_by_formats_v2(content_pages)
            type_classified_pages = classify_page_text_by_types(cleaned_pages)
            extractable_data = clean_text_by_types_v2(type_classified_pages)

            extracted_data = extract_data_v2(extractable_data)

            for node in get_node_children(extracted_data, input_node_route):
                output_obtained.append(node.value)

            self.assertEqual(output_expected, output_obtained)

    # Create a test suite and add test cases
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExtraction)
    result = unittest.TextTestRunner(verbosity=0).run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    run_tests()
