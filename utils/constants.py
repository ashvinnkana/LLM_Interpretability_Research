from models.anthropic_client import ANTHROPIC
from models.node import Node
from models.groq_client import GROQ
from models.openai_client import OpenAIGPT

sample_score = {
    'llama3-70b-8192': {'unstructured': [{'rouge1': 0.48275862068965514, 'rougeL': 0.3310344827586207}, {'rouge1': 0.4507042253521127, 'rougeL': 0.295774647887324}], 'json-structured-v1': [{'rouge1': 0.481203007518797, 'rougeL': 0.3157894736842105}, {'rouge1': 0.43200000000000005, 'rougeL': 0.336}], 'html-structured-v1': [{'rouge1': 0.3364485981308411, 'rougeL': 0.26168224299065423}, {'rouge1': 0.32727272727272727, 'rougeL': 0.2545454545454545}], 'toml-structured-v2': [{'rouge1': 0.5306122448979592, 'rougeL': 0.40816326530612246}, {'rouge1': 0.5714285714285715, 'rougeL': 0.30075187969924805}], 'md-structured-v2': [{'rouge1': 0.5384615384615384, 'rougeL': 0.3384615384615384}, {'rouge1': 0.40983606557377045, 'rougeL': 0.2786885245901639}], 'json-structured-v2': [{'rouge1': 0.5985401459854014, 'rougeL': 0.35036496350364965}, {'rouge1': 0.5671641791044777, 'rougeL': 0.3582089552238806}], 'html-structured-v2': [{'rouge1': 0.5384615384615384, 'rougeL': 0.3384615384615384}, {'rouge1': 0.44247787610619466, 'rougeL': 0.37168141592920356}]},
    'mixtral-8x7b-32768': {'unstructured': [{'rouge1': 0.4214876033057851, 'rougeL': 0.2809917355371901}, {'rouge1': 0.40449438202247195, 'rougeL': 0.30337078651685395}], 'json-structured-v1': [{'rouge1': 0.45833333333333337, 'rougeL': 0.2777777777777778}, {'rouge1': 0.4525547445255475, 'rougeL': 0.2773722627737226}], 'html-structured-v1': [{'rouge1': 0.4, 'rougeL': 0.36363636363636365}, {'rouge1': 0.4, 'rougeL': 0.36363636363636365}], 'toml-structured-v2': [{'rouge1': 0.6103896103896104, 'rougeL': 0.3246753246753247}, {'rouge1': 0.5882352941176471, 'rougeL': 0.42352941176470593}], 'md-structured-v2': [{'rouge1': 0.5730994152046783, 'rougeL': 0.4093567251461988}, {'rouge1': 0.6022727272727272, 'rougeL': 0.39772727272727276}], 'json-structured-v2': [{'rouge1': 0.5930232558139534, 'rougeL': 0.37209302325581395}, {'rouge1': 0.6395348837209303, 'rougeL': 0.38372093023255816}], 'html-structured-v2': [{'rouge1': 0.4711538461538462, 'rougeL': 0.2692307692307692}, {'rouge1': 0.6357615894039734, 'rougeL': 0.3841059602649006}]},
    'gemma-7b-it': {'unstructured': [{'rouge1': 0.3220338983050847, 'rougeL': 0.288135593220339}, {'rouge1': 0.39062500000000006, 'rougeL': 0.31249999999999994}], 'json-structured-v1': [{'rouge1': 0.417391304347826, 'rougeL': 0.31304347826086953}, {'rouge1': 0.40310077519379844, 'rougeL': 0.31007751937984496}], 'html-structured-v1': [{'rouge1': 0.33962264150943394, 'rougeL': 0.33962264150943394}, {'rouge1': 0.36363636363636365, 'rougeL': 0.34545454545454546}], 'toml-structured-v2': [{'rouge1': 0.40625000000000006, 'rougeL': 0.31249999999999994}, {'rouge1': 0.40310077519379844, 'rougeL': 0.32558139534883723}], 'md-structured-v2': [{'rouge1': 0.4107142857142857, 'rougeL': 0.39285714285714285}, {'rouge1': 0.4107142857142857, 'rougeL': 0.39285714285714285}], 'json-structured-v2': [{'rouge1': 0.5072463768115942, 'rougeL': 0.3623188405797102}, {'rouge1': 0.4144144144144144, 'rougeL': 0.3783783783783784}], 'html-structured-v2': [{'rouge1': 0.4107142857142857, 'rougeL': 0.39285714285714285}, {'rouge1': 0.4144144144144144, 'rougeL': 0.39639639639639634}]},
    'claude-3-opus-20240229': {'unstructured': [{'rouge1': 0.4461538461538461, 'rougeL': 0.35384615384615387}, {'rouge1': 0.421875, 'rougeL': 0.28125}], 'json-structured-v1': [{'rouge1': 0.46153846153846156, 'rougeL': 0.358974358974359}, {'rouge1': 0.42975206611570244, 'rougeL': 0.38016528925619836}], 'html-structured-v1': [{'rouge1': 0.4347826086956521, 'rougeL': 0.39999999999999997}, {'rouge1': 0.4390243902439025, 'rougeL': 0.3902439024390244}], 'toml-structured-v2': [{'rouge1': 0.528, 'rougeL': 0.384}, {'rouge1': 0.5859872611464968, 'rougeL': 0.45859872611464964}], 'md-structured-v2': [{'rouge1': 0.60431654676259, 'rougeL': 0.4748201438848921}, {'rouge1': 0.5396825396825397, 'rougeL': 0.4285714285714286}], 'json-structured-v2': [{'rouge1': 0.60431654676259, 'rougeL': 0.5179856115107914}, {'rouge1': 0.5605095541401273, 'rougeL': 0.4331210191082802}], 'html-structured-v2': [{'rouge1': 0.5161290322580645, 'rougeL': 0.4032258064516129}, {'rouge1': 0.5616438356164384, 'rougeL': 0.47945205479452063}]},
    'gpt-3.5-turbo': {'unstructured': [{'rouge1': 0.0, 'rougeL': 0}, {'rouge1': 0.0, 'rougeL': 0}], 'json-structured-v1': [{'rouge1': 0.0, 'rougeL': 0}, {'rouge1': 0.0, 'rougeL': 0}], 'html-structured-v1': [{'rouge1': 0.0, 'rougeL': 0}, {'rouge1': 0.0, 'rougeL': 0}], 'toml-structured-v2': [{'rouge1': 0.0, 'rougeL': 0}, {'rouge1': 0.0, 'rougeL': 0}], 'md-structured-v2': [{'rouge1': 0.0, 'rougeL': 0}, {'rouge1': 0.0, 'rougeL': 0}], 'json-structured-v2': [{'rouge1': 0.0, 'rougeL': 0}, {'rouge1': 0.0, 'rougeL': 0}], 'html-structured-v2': [{'rouge1': 0.0, 'rougeL': 0}, {'rouge1': 0.0, 'rougeL': 0}]}
}

# outperforms all known language models for Australian law (perplexity : 8.01)
aus_legal_llm = 'umarbutler/open-australian-legal-llm'

groq_interface = GROQ()
openai_interface = OpenAIGPT()
anthropic_interface = ANTHROPIC()

non_legal_llm_list = [
    {'model_id': 'llama3-70b-8192',
     'developer': 'Meta',
     'client': groq_interface},
    {'model_id': 'mixtral-8x7b-32768',
     'developer': 'Mistral',
     'client': groq_interface},
    {'model_id': 'gemma-7b-it',
     'developer': 'Google',
     'client': groq_interface},
    {'model_id': 'claude-3-opus-20240229',
     'developer': 'Anthropic',
     'client': anthropic_interface},
    {'model_id': 'gpt-3.5-turbo',
     'developer': 'OpenAI',
     'client': openai_interface}
]

fetch_docs_count = 5
upsert_batch_size = 5

header_estimated_size = 4
footer_estimated_size = 4
header_footer_occurrence_accept_threshold = 2
header_tag = 'header'
footer_tag = 'footer'

header_footer_estimated_size_v2 = 6
header_footer_occurrence_accept_threshold_v2 = 1

# keep it lower for small unstructured file
chunk_window_size = 1
chunk_token_limit = 1000

unstructured_tag = 'UNSTRUCTURED'
json_structured_tag = 'JSON_STRUCTURED'
html_structured_tag = 'HTML_STRUCTURED'

nltk_resource_packages = [
    'stopwords',
    'punkt'
]

embedder = 'dwzhu/e5-base-4k'
pinecone_cloud = 'aws'
pinecone_region = 'us-east-1'
pinecone_dimension = 768

latest_head_index = 0

root_node_v2 = Node('root', 'structured_data', '')

docs_language = 'english'
