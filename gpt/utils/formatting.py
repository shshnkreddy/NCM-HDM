import re 

def is_importance_format(input_string):
    pattern = r'^Importance: \d+(\.\d+)?$'
    return bool(re.match(pattern, input_string.strip()))    
