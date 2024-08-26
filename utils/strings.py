from data.legal_llm_responses.legal_llm_responses import legal_llm_responses

nlt_source_path = 'tokenizers/{}'
csv_results_path = 'results/temp/{}_{}.csv'
structured_data_path = 'data/{}/{}_{}_{}.{}'
samples_docs_path = 'data/samples/{}_docs_retrieved.json'
logging_config_file = 'logging.conf'

# vectordb index names
extraction_v0_index = 'v0-extracted-data-index'
extraction_v1_index = 'v1-extracted-data-index'
extraction_v2_index = 'v2-extracted-data-index'

unstructured_pdf_paths = [
    './data/unstructured_data/Limitation_Act_2005.pdf'
]

questions = [
    {'id': 'question_01',
     'query': 'Which section of the law in WA specifies the limitation period for commencing an action to recover '
              'land since the trespassing accrued?',
     'ref_answer': legal_llm_responses['question_01']}
]

header_footer_level_setup_string = '{}->{}'
classify_level_string = '{}-{}-{}-{}'

unstructured_llm_message = (
    'SYSTEM MESSAGE:\n'
    'You are a helpful {} assistant that answers questions using the context provided.\n\n'
    'CONTEXT:\n{}')

unstructured_question = 'QUESTION: {} (Answer in 150 Words)'

md_llm_message = (
    '# SYSTEM MESSAGE:\n'
    'You are a helpful {} assistant that answers questions using the context provided.\n\n'
    '# CONTEXT:\n{}')

md_question = '# QUESTION:\n{} (Answer in 150 Words)'

toml_llm_message = (
    '["SYSTEM MESSAGE"]\n'
    'title = "System Message"\n'
    'description = "You are a helpful {} assistant that answers questions using the [context] provided."\n\n'
    '{}')

toml_question = ('["QUESTION"]\n'
                 'title = "Question"\n'
                 'description = "{} (Answer in 150 Words)"\n\n')

html_llm_message = (
    '<SYSTEM-MESSAGE>'
    'You are a helpful {} assistant that answers questions using the context provided.'
    '</SYSTEM-MESSAGE>'
    '<CONTEXT>'
    '{}'
    '</CONTEXT>')

html_question = '<QUESTION>{} (Answer in 150 Words)</QUESTION>'

json_llm_message = (
    '{{'
    '"SYSTEM-MESSAGE":"You are a helpful {} assistant that answers questions using the context provided.",'
    '"CONTEXT":{}'
    '}}')

json_question = '{{"QUESTION":"{} (Answer in 150 Words)"}}'
