import json 
import pandas as pd
import sys
sys.path.append('../')
from ast import literal_eval
import numpy as np
from Memory.base import *
import openai 
from Prompting.prompt import * 
import re
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from key import OPEN_AI_KEY
import os

client = openai.OpenAI(api_key = OPEN_AI_KEY)
model = 'gpt-3.5-turbo'  

file_path = 'incontext_template.txt'
with open(file_path, 'r') as file:
    incontext_template = file.read()

def extract_justification_and_response(text):
        # Define regular expression patterns to match the justification and response
        justification_pattern = r"Justification:(.*?)Response:"
        response_pattern = r"Response:\s*(YES|NO)"

        # Use regular expressions to extract the justification and response
        justification_match = re.search(justification_pattern, text, re.DOTALL)
        response_match = re.search(response_pattern, text)

        # Check if both patterns are found
        if justification_match and response_match:
            # Extract the justification and response from the matches
            justification = justification_match.group(1).strip()
            response = response_match.group(1).strip()
            return justification, response
        else:
            return None, None

def prompt_classification(input, back_up_response):
    sys = None
    prompt = incontext_template
    prompt += f"{input}\n"
    prompt += "\nPlease answer the question in the following format:\nJustification: {a comprehensive justification for your predictions with citations to emails in the history to ensure reliability}\nResponse: {your response YES/NO}"
    # print(prompt)
    output = prompt_open_ai(client, model, prompt, None)
    unfiltered_response = output
    justification, response = extract_justification_and_response(output.strip())
    if(response is None):
        response = back_up_response
        justification = ''
        is_backup = True
    else:
        is_backup = False
        
    return prompt, response, justification, is_backup, unfiltered_response

def extract_response(text):
        text = text.replace(" ", '')
        pattern = r'^Response:(YES|NO)$'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None

def find_back_up(text):
    text = text.strip()
    pattern = r'Response:(YES|NO)'
    match = re.search(pattern, text)
    
    if match:
        response_value = match.group(1)
        return response_value.upper()
    else:
        return None
    
def save_lists(llm_outputs_mem, is_backups_mem, llm_justs, llm_unfiltered, prompts):
    data = {
        "llm_outputs_mem": llm_outputs_mem,
        "is_backups_mem": is_backups_mem,
        "llm_justs": llm_justs,
        "llm_unfiltered": llm_unfiltered,
        "prompts": prompts
    }
    with open(f'{model}.json', 'w') as file:
        json.dump(data, file, indent=4)

def read_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():
    df = pd.read_csv('embedded_emails.csv')
    df['current_embedding'] = df['current_embedding'].apply(lambda x: np.array(literal_eval(x.strip())))
    df['current'] = df['current'].apply(lambda x: x.strip('Email:\n'))
    df = df.sort_values(by=['phase','trial'])
    ids = df['Mturk_id'].unique()

    memories = {}
    for id in ids:
        memories[id] = Memory(None, None, None)

    histories = []
    bad_histories = []
    back_up_responses = []
    i = 0
    set = []
    for index, row in df.iterrows():
        id = row['Mturk_id']
        current = row['current']
        response = row['output']
        embedding = row['current_embedding']
        data = current+f'\nResponse:{response}'
        #Retrieval
        ret_nodes, _, bad_nodes, _  = memories[id].retrieve(query=current, t=i, n=5, query_embedding=embedding, alpha_t=0.0, ret_least_score=True)
        
        #Store History
        concat_history = ''
        j = 1
        for node in ret_nodes:
            concat_history += f'Email: {j}\n'
            concat_history += f'{node.data}\n\n'
            j += 1
        histories.append(concat_history)

        concat_bad = ''
        for node in bad_nodes:
            concat_bad += f'{node.data}\n\n'
        bad_histories.append(concat_bad)

        #Store Current Email
        memories[id].insert(None, i, data, embedding)
        i+=1 

        #test on emails 11-60
        if(memories[id].get_n() <= 10):
            set.append('train')
        else:
            set.append('test')

    df['history'] = histories
    df['set'] = set

    df_test = df[df['set']=='test']

    fpath = f'/common/home/users/s/shashankc/code/Phishing/LLM_Prompting/{model}.json'
    if(os.path.exists(fpath)):
        data = read_data_from_json(fpath)
        llm_outputs_mem = data["llm_outputs_mem"]
        is_backups_mem = data["is_backups_mem"]
        llm_justs = data["llm_justs"]
        llm_unfiltered = data["llm_unfiltered"]
        prompts = data["prompts"]
        print('Loaded from Disk')

    else:
        llm_outputs_mem = []
        is_backups_mem = []
        llm_justs = []
        llm_unfiltered = []
        prompts = []

    
    save_interval = 100  # Define the save interval
    print_interval = 100   # Define the print progress interval
    for i in range(len(prompts), len(df_test)):
        start = time.time()
        if(i % save_interval == 0 and i != 0):  # Save lists periodically
            save_lists(llm_outputs_mem, is_backups_mem, llm_justs, llm_unfiltered, prompts)
            print("Lists saved.", flush=True)
        
        row = df_test.iloc[i]
        current = row['current']
        history = row['history']
        _, llm_output_mem, llm_just, is_backup_mem, ll_unf = prompt_classification(f'##History Emails and Responses:\n{history}##Current Email and Response\n{current}', "BACKUP")
        is_backups_mem.append(is_backup_mem)
        llm_outputs_mem.append(llm_output_mem)
        llm_justs.append(llm_just)
        llm_unfiltered.append(ll_unf)
        prompts.append(f'##History Emails and Responses:\n{history}##Current Email and Response\n{current}')
        end = time.time()
        
        if i % print_interval == 0:
            print(f'Progress: {i}/{len(df_test)}')
            print(f'Time Taken: {end-start}')
            
    # Save the remaining lists
    save_lists(llm_outputs_mem, is_backups_mem, llm_justs, llm_unfiltered, prompts)
    print("Final lists saved.")
        
    df_test['llm_outputs_mem'] = llm_outputs_mem
    df_test['is_backups_mem'] = is_backups_mem
    df_test['justification'] = llm_justs
    df_test.to_csv(f'{model}.csv', index=False)

    #Number of Backups
    print('Memory:', df_test['is_backups_mem'].sum() / len(df_test))
        
    gt = df_test['email_type'].apply(lambda x: "YES" if x=="phishing" else "NO")
    print(gt)

    #Accuracy of human 
    print((df_test['output'] == gt).sum() / len(gt))

    #Accuracy LLM with Memory 
    print((df_test['llm_outputs_mem'] == gt).sum() / len(gt))
    
    
    #Accuracy of LLM with memory w.r.t human 
    print('Accuracy of LLM w.r.t Human')
    print((df_test['llm_outputs_mem'] == df_test['output']).sum() / len(df_test['output']))

    # cf = confusion_matrix(df_test['output'], df_test['llm_outputs_mem'], normalize='all', labels = ['NO', 'YES'])
    # #Confusion Matrix with Memory
    # disp = ConfusionMatrixDisplay(cf, display_labels = ['NO', 'YES'])
    # disp.plot()
    # plt.title(f'{model}')
    # plt.savefig(f'{model}.png')

if __name__ == "__main__":
    main()


