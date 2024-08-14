alpha_characters_only = r'[^a-zA-Z\s]'
alpha_characters_only_v2 = r'[^a-zA-Z\s]|[ivxlcdm]'
no_special_characters_v2 = r'[^\w\s.,!?$%()[\]{}-]'
start_with_bullet_pattern = r'^\s*[\u2022\-\*\d+\.\)]|^\s*\(\d+\)|^\s*\([a-zA-Z]+\)|^\s*[a-zA-Z]\.' + \
              r'|^\s*[\u2023\u2219\u25E6\u25AA\u25CF\u25CB\u25A0-\u25FF]'
ends_with_special = r'[^a-zA-Z0-9\s,]$'
newline_before_colon = r'\n(?=:)'
extra_space_btw_words = r'(?<!\n) +'
multiple_consecutive_newlines = r'\n\n+'
bullet_patterns = {
        '•': r'^\s*([\u2022])\s*(.*)',
        '-': r'^\s*([-])\s*(.*)',
        '*': r'^\s*([*])\s*(.*)',
        'numbered_lettered': r'^\s*(\d+)\.([a-zA-Z])\s*(.*)',  # Capture number followed by letter and dot
        'numbered_numbered': r'^\s*(\d+)\.(\d+)\s*(.*)',  # Capture number followed by another number and dot
        'numbered': r'^\s*(\d+)\.\s*(.*)',  # Capture number before the dot and the text after
        'lettered': r'^\s*\(([a-zA-Z])\)\s*(.*)',  # Capture letter inside parentheses and the text after
        'single_letter': r'^\s*([a-zA-Z])\.\s*(.*)',  # Capture single letter followed by dot and the text after
        # Capture bullet symbol and the text after
        'bullet_symbol': r'^\s*([\u2023\u2219\u25E6\u25AA\u25CF\u25CB\u25A0-\u25FF])\s*(.*)',
        'mixed_numbered': r'^\s*(\d+[a-zA-Z]?)\.\s*(.*)',  # Capture number followed by optional letter and dot
        # Capture number followed by optional letter and text inside parentheses
        'mixed_numbered_parentheses': r'^\s*(\d+[a-zA-Z]?)\s*\((.*?)\)\s*(.*)',
        'mixed_numbered_dot': r'^\s*(\d+[a-zA-Z]?)\s*\.\s*(.*)',  # Capture number followed by optional letter and dot
        'parentheses_number': r'^\s*\((\d+[a-zA-Z]?)\)\s*(.*)',  # Capture number or letter in parentheses
}
number_parenthese_bullet = r'^\((\d+)\)$'
number_with_number_parenthese_bullet = r'^(\d+)\((\d+)\)$'
number_alpa_char_bullet = r'^(\d+)([a-z])$'
float_number_bullet = r'^(\d+)\.(\d+)$'
float_alpha_bullet = r'^(\d+)\.([a-z])$'
simple_number_bullet = r'^(\d+)$'
simple_alpha_bullet = r'^([a-z])$'
part_division_headings = r"(?:Division|Part) \d+ — "
function_words_ending = (r'\b(?:\w+\s)*(or|is|are|were|was|a|and|these|where|but|nor|yet|the|an|some|that|am|has|have'
                         r'|do|does|had|in|on|at|with|it|they|this|then|here|there|why)\b[.!?]?')
ending_with_numbers = r'\d+$'
