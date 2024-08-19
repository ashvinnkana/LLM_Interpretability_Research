nlt_source_path = 'tokenizers/{}'
csv_results_path = 'results/{}_{}.csv'
structured_data_path = 'data/{}/{}_{}_{}.{}'
samples_docs_path = 'data/samples/{}_docs_retrieved.json'
logging_config_file = 'logging.conf'

# vectordb index names
unstructured_0_index = 'v0-unstructured-v0-extract-index'
structured_JSON_1_index = 'v0-json-structured-v1-extract-index'
structured_v2_index = 'structured-v2-extract-index'
structured_HTML_1_index = 'v0-html-structured-v1-extract-index'

unstructured_pdf_paths = [
    './data/unstructured_data/Limitation_Act_2005.pdf'
]

questions = [
    {'id': 'question_01',
     'query': 'Which section of the law in WA specifies the limitation period for initiating an action to recover '
              'land that has been trespassed?',
     'ref_answer': 'Section 19(1) of the Limitation Act 2005 specifies that the limitation period for initiating an '
                   'action to recover land is 12 years from the date when the cause of action first accrued. This '
                   'means that if the land has been trespassed, the rightful owner must take legal action within 12 '
                   'years, or they may lose the right to recover the property. However, section 19(2) states an '
                   'exception to legal proceedings initiated by the Crown or any individual asserting rights through '
                   'the Crown.'}
]

header_footer_level_setup_string = '{}->{}'
classify_level_string = '{}-{}-{}-{}'

unstructured_llm_message = (
    'SYSTEM MESSAGE:\n'
    'You are a helpful assistant that answers questions in two sentences about {} using the context provided.\n\n'
    'CONTEXT:\n{}')

unstructured_question = 'QUESTION: {}'

md_llm_message = (
    '# SYSTEM MESSAGE:\n'
    'You are a helpful assistant that answers questions in two sentences about {} using the context provided.\n\n'
    '# CONTEXT:\n{}')

md_question = '# QUESTION:\n{}'

toml_llm_message = (
    '["SYSTEM MESSAGE"]\n'
    'title = "System Message"\n'
    'description = "You are a helpful assistant that answers questions in two sentences about {} using the [context] '
    'provided."\n\n'
    '{}')

toml_question = ('["QUESTION"]\n'
                 'title = "Question"\n'
                 'description = "{}"\n\n')

html_llm_message = (
    '<SYSTEM-MESSAGE>'
    'You are a helpful assistant that answers questions in two sentences about {} using the context provided.'
    '</SYSTEM-MESSAGE>'
    '<CONTEXT>'
    '<ul>{}</ul>'
    '</CONTEXT>')

html_question = '<QUESTION>{}</QUESTION>'

json_llm_message = (
    '{{'
    '"SYSTEM-MESSAGE":"You are a helpful assistant that answers questions in two sentences about {} using the context '
    'provided",'
    '"CONTEXT":{}'
    '}}')

json_question = '{{"QUESTION":"{}"}}'
