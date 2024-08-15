import unittest

from scripts.extract_data import extract_v2
from utils.functions import get_node_children


def run_tests():
    class TestExtraction(unittest.TestCase):
        def test_with_extracted_headings(self):
            input_pdf = './../data/unstructured_data/Limitation_Act_2005.pdf'
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

            extracted_data = extract_v2(input_pdf)
            for node in get_node_children(extracted_data, input_node_route):
                output_obtained.append(node.value)

            self.assertEqual(output_expected, output_obtained)

    # Create a test suite and add test cases
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExtraction)
    result = unittest.TextTestRunner(verbosity=0).run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    run_tests()
