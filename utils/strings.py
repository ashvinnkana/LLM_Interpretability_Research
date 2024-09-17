from data.legal_llm_responses.V2_1_legal_llm_responses import V2_1_legal_llm_responses

nlt_source_path = 'tokenizers/{}'
csv_results_path = 'results/temp/{}_{}.csv'
structured_data_path = 'data/{}/{}_{}_{}.{}'
samples_docs_path = 'data/others/{}_docs_retrieved.json'
logging_config_file = 'logging.conf'

# vectordb index names
extraction_v0_index = 'v0-extracted-data-index'
extraction_v1_index = 'v1-extracted-data-index'
extraction_v2_index = 'v2-extracted-data-index'
extraction_v2_1_index = 'v2-1-extracted-data-index'

unstructured_pdf_paths = [
    './data/unstructured_data/Limitation_Act_2005.pdf',
    './data/unstructured_data/Criminal_Code_Act_Compilation_Act_1913.pdf',
    './data/unstructured_data/Adoption_Act_1994.pdf',
    './data/unstructured_data/Births_Deaths_and_Marriages_Registration_Act_1998.pdf',
    './data/unstructured_data/Cat_Act_2011.pdf',
    './data/unstructured_data/Dog_Act_1976.pdf',
    './data/unstructured_data/Family_Violence_Legislation_Reform_Act_2020.pdf',
    './data/unstructured_data/Misuse_Of_Drugs_Act_1981.pdf',
    './data/unstructured_data/Residential_Tenancies_Act_1987.pdf',
    './data/unstructured_data/Road_Traffic_(Vehicles)_Act_2012.pdf',
    './data/unstructured_data/Surrogacy_Act_2008.pdf'
]

questions = [
    {'id': 'question_01',
     'query': 'Which section of the law in WA specifies the limitation period for commencing an action to recover '
              'land since the trespassing accrued?',
     'ref_answer': V2_1_legal_llm_responses['question_01']},
    {'id': 'question_02',
     'query': 'Which section of the law in WA specifies the penalty for Common assault?',
     'ref_answer': V2_1_legal_llm_responses['question_02']},
    {'id': 'question_03',
     'query': 'Which section of the law in WA specifies the imprisonment period for an unlawful assault causing death?',
     'ref_answer': V2_1_legal_llm_responses['question_03']},
    {'id': 'question_04',
     'query': 'Which section of the law in WA states the penalty for Stealing?',
     'ref_answer': V2_1_legal_llm_responses['question_04']},
    {'id': 'question_05',
     'query': 'Which section of the Criminal Code in WA states about Fraud?',
     'ref_answer': V2_1_legal_llm_responses['question_05']},
    {'id': 'question_06',
     'query': 'Which section of the Criminal Code states the unlawful killing of another person under circumstances '
              'to not constitute murder?',
     'ref_answer': V2_1_legal_llm_responses['question_06']},
    {'id': 'question_07',
     'query': 'Which section of the Criminal Code can the defence of accident be found?',
     'ref_answer': V2_1_legal_llm_responses['question_07']},
    {'id': 'question_08',
     'query': 'What does the term ‘applies forces’ include under Section 222 in Criminal Code Act of WA?',
     'ref_answer': V2_1_legal_llm_responses['question_08']},
    {'id': 'question_09',
     'query': 'Which section of the criminal code states the standard of proof required in criminal cases?',
     'ref_answer': V2_1_legal_llm_responses['question_09']},
    {'id': 'question_10',
     'query': 'Which section of the law in WA specifies the imprisonment period for an aggravated home burglary offence?',
     'ref_answer': V2_1_legal_llm_responses['question_10']},
    {'id': 'question_11',
     'query': 'Who are the restricted persons for change of name in birth, death or marriage registration, according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_11']},
    {'id': 'question_12',
     'query': 'What is the section of the law in WA that states the restriction on changing a child’s name in birth registration?',
     'ref_answer': V2_1_legal_llm_responses['question_12']},
    {'id': 'question_13',
     'query': 'What is the penalty in WA if the responsible person in relation to the birth of a child does not give the Registrar notice of the birth?',
     'ref_answer': V2_1_legal_llm_responses['question_13']},
    {'id': 'question_14',
     'query': 'Which section of the law in WA states the documents to include to register a marriage?',
     'ref_answer': V2_1_legal_llm_responses['question_14']},
    {'id': 'question_15',
     'query': 'Which section of the law in WA states the limit for an adult to register changes of their name?',
     'ref_answer': V2_1_legal_llm_responses['question_15']},
    {'id': 'question_16',
     'query': 'Is Cannabis listed as a prohibited drug by the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_16']},
    {'id': 'question_17',
     'query': 'Which section of the law in WA states about the offence of possessing prohibited drugs with the intent to sell or supply to another?',
     'ref_answer': V2_1_legal_llm_responses['question_17']},
    {'id': 'question_18',
     'query': 'What is the penalty for a person selling drug paraphernalia to an adult, according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_18']},
    {'id': 'question_19',
     'query': 'Which section of the Family Violence law in WA specifies the penalty for deprivation of liberty?',
     'ref_answer': V2_1_legal_llm_responses['question_19']},
    {'id': 'question_20',
     'query': 'What is the penalty for suffocation and strangulation by the Family Violence law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_20']},
    {'id': 'question_21',
     'query': 'What is the penalty in WA law, if a person persistently engages in family violence?',
     'ref_answer': V2_1_legal_llm_responses['question_21']},
    {'id': 'question_22',
     'query': 'Which section of the law in WA applies, if 2 or more tenants who are parties to a residential tenancy agreement die?',
     'ref_answer': V2_1_legal_llm_responses['question_22']},
    {'id': 'question_23',
     'query': 'What are the requirements that a notice of termination by lessor must contain according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_23']},
    {'id': 'question_24',
     'query': 'Which section of the law in WA mentions the termination of social housing tenancy agreement due to objectionable behaviour?',
     'ref_answer': V2_1_legal_llm_responses['question_24']},
    {'id': 'question_25',
     'query': 'When does a person who uses the vehicle on a road commit an offence, according to the Road Traffic law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_25']},
    {'id': 'question_26',
     'query': 'What are the circumstances the CEO may cancel the licence of a vehicle, according to the Road Traffic law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_26']},
    {'id': 'question_27',
     'query': 'Which section of the law in WA applies to vehicles with authorization to continue journey, if there are minor risk breaches?',
     'ref_answer': V2_1_legal_llm_responses['question_27']},
    {'id': 'question_28',
     'query': 'Which section of the Road Traffic Law in WA specifies the duty of a driver?',
     'ref_answer': V2_1_legal_llm_responses['question_28']},
    {'id': 'question_29',
     'query': 'What does a person have to prove to have the benefit of the reasonable steps defence, according to the Road Traffic law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_29']},
    {'id': 'question_30',
     'query': 'What is the penalty by the law in WA, for someone who enters a surrogacy arrangement that is for reward?',
     'ref_answer': V2_1_legal_llm_responses['question_30']},
    {'id': 'question_31',
     'query': 'What are the requirements for surrogacy arrangement to be approved, according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_31']},
    {'id': 'question_32',
     'query': 'According to the Surrogacy law in WA, which section states, who has access to registration of birth of a person?',
     'ref_answer': V2_1_legal_llm_responses['question_32']},
    {'id': 'question_33',
     'query': 'According to the Surrogacy law in WA, which section specifies, what must the court have regard to before making an order to change the child’s name?',
     'ref_answer': V2_1_legal_llm_responses['question_33']},
    {'id': 'question_34',
     'query': 'Which section of the law in WA specifies, when can an application be made for a parentage order?',
     'ref_answer': V2_1_legal_llm_responses['question_34']},
    {'id': 'question_35',
     'query': 'Which section of the law in WA specifies the procedure to register a dog?',
     'ref_answer': V2_1_legal_llm_responses['question_35']},
    {'id': 'question_36',
     'query': 'According to the law in WA, what is the time frame that the new registered owner of a dog upon a change of ownership notify the local government about the name and residential address change?',
     'ref_answer': V2_1_legal_llm_responses['question_36']},
    {'id': 'question_37',
     'query': 'Which section of the law in WA specifies about the conditions for a dog registration tag?',
     'ref_answer': V2_1_legal_llm_responses['question_37']},
    {'id': 'question_38',
     'query': 'What is the penalty by the law in WA, of offences relating to dog registration?',
     'ref_answer': V2_1_legal_llm_responses['question_38']},
    {'id': 'question_39',
     'query': 'Who has the power to seize stray dogs, according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_39']},
    {'id': 'question_40',
     'query': 'Which section of the law in WA specifies about the defences against a dog attack?',
     'ref_answer': V2_1_legal_llm_responses['question_40']},
    {'id': 'question_41',
     'query': 'In what situation can the registration of a cat be cancelled, according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_41']},
    {'id': 'question_42',
     'query': 'According to the law in WA, what is the age limit that a cat must be microchipped?',
     'ref_answer': V2_1_legal_llm_responses['question_42']},
    {'id': 'question_43',
     'query': 'What is the penalty for breeding cats without being an approved cat breeder by the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_43']},
    {'id': 'question_44',
     'query': 'Which section of the law in WA specifies when can an authorized person cause a cat to be destroyed?',
     'ref_answer': V2_1_legal_llm_responses['question_44']},
    {'id': 'question_45',
     'query': 'What cost is the cat owner liable to pay to cat management facility according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_45']},
    {'id': 'question_46',
     'query': 'Which section of the law in WA specifies the penalty for unauthorized adoption services?',
     'ref_answer': V2_1_legal_llm_responses['question_46']},
    {'id': 'question_47',
     'query': 'Whose consent is required to adopt a child, according to the law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_47']},
    {'id': 'question_48',
     'query': 'According to the law in WA, what is the time frame for the consent to a child’s adoption to be given after the child is born?',
     'ref_answer': V2_1_legal_llm_responses['question_48']},
    {'id': 'question_49',
     'query': 'When does parental responsibility cease according to the Adoption law in WA?',
     'ref_answer': V2_1_legal_llm_responses['question_49']},
    {'id': 'question_50',
     'query': 'Which section of the law in WA specifies who may adopt a child?',
     'ref_answer': V2_1_legal_llm_responses['question_50']},
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

code_llm_message = (
    'SYSTEM_MESSAGE = "You are a helpful {} assistant that answers questions using the context provided."'
    'class CONTEXT:'
    '\tdef __init__(self, heading_, content_):'
    '\t\tself.heading = heading_'
    '\t\tself.content = content_'
    '\n{}'
)

code_question = 'QUESTION = "{} (Answer in 150 Words)"'
