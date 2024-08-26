from utils import strings
from models.pinecone_client import PINECONE
from utils.functions import v1_json_process_docs, v2_json_process_docs, v2_markdown_process_docs, \
 v2_custom3_process_docs
from utils.functions import v1_html_process_docs, v2_html_process_docs, unstruct_process_docs
from utils.functions import v2_toml_process_docs, v2_custom1_process_docs, v2_custom2_process_docs

v0_extraction_vectordb = PINECONE(strings.extraction_v0_index)
v1_extraction_vectordb = PINECONE(strings.extraction_v1_index)
v2_extraction_vectordb = PINECONE(strings.extraction_v2_index)

stashed_format_lists = [
    {'id': 'v1-json',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v1_json_process_docs,
     'vector_db': v1_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v1-html',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v1_html_process_docs,
     'vector_db': v1_extraction_vectordb,
     'doc_count': 4},
]

format_lists = [
    {'id': 'v0-unstruct',
     'query_str': strings.unstructured_question,
     'llm_msg_str': strings.unstructured_llm_message,
     'get_docs_func': unstruct_process_docs,
     'vector_db': v0_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-toml',
     'query_str': strings.toml_question,
     'llm_msg_str': strings.toml_llm_message,
     'get_docs_func': v2_toml_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 2},
    {'id': 'v2-md',
     'query_str': strings.md_question,
     'llm_msg_str': strings.md_llm_message,
     'get_docs_func': v2_markdown_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-json',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v2_json_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-html',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v2_html_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-custom1',
     'query_str': strings.html_question,
     'llm_msg_str': strings.html_llm_message,
     'get_docs_func': v2_custom1_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
    {'id': 'v2-custom2',
     'query_str': strings.json_question,
     'llm_msg_str': strings.json_llm_message,
     'get_docs_func': v2_custom2_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 4},
     {'id': 'v2-custom3',
     'query_str': strings.toml_question,
     'llm_msg_str': strings.toml_llm_message,
     'get_docs_func': v2_custom3_process_docs,
     'vector_db': v2_extraction_vectordb,
     'doc_count': 2}
]
