from models.node import Node
from models.groq_client import GROQ
from models.openai_client import OpenAIGPT


# outperforms all known language models for Australian law (perplexity : 8.01)
aus_legal_llm = 'umarbutler/open-australian-legal-llm'

groq_interface = GROQ()
openai_interface = OpenAIGPT()

non_legal_llm_list = [
    {'model_id': 'llama3-70b-8192',
     'developer': 'Meta',
     'client': groq_interface},
    {'model_id': 'gemma-7b-it',
     'developer': 'Google',
     'client': groq_interface},
    {'model_id': 'gpt-3.5-turbo',
     'developer': 'OpenAI',
     'client': openai_interface},
]

fetch_docs_count = 5

header_estimated_size = 4
footer_estimated_size = 4
header_footer_occurrence_accept_threshold = 2
header_tag = 'header'
footer_tag = 'footer'

header_footer_estimated_size_v2 = 6
header_footer_occurrence_accept_threshold_v2 = 1

# keep it lower for small unstructured file
chunk_window_size = 1
chunk_token_limit = 1300

unstructured_tag = 'UNSTRUCTURED'
json_structured_tag = 'JSON_STRUCTURED'
html_structured_tag = 'HTML_STRUCTURED'

nltk_resource_packages = [
    'punkt'
]

embedder = 'dwzhu/e5-base-4k'
pinecone_cloud = 'aws'
pinecone_region = 'us-east-1'
pinecone_dimension = 768

latest_head_index = 0

root_node_v2 = Node('root', 'structured_data', '')

