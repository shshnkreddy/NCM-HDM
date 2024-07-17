from key import OPEN_AI_KEY
import openai 
import pandas as pd
import sys
sys.path.append('../')
import json
import sys
sys.path.append('../')
from Memory.base import *
from Prompting.prompt import * 
import re
import os
import time

client = openai.OpenAI(api_key = OPEN_AI_KEY)
model = 'gpt-3.5-turbo'  

#Load the template
file_path = 'incontext_template_IAG'
# Open the file in read mode
with open(file_path, 'r') as file:
    # Read the entire content of the file and store it in a string variable
    incontext_template = file.read()

def read_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_lists(llm_outputs_mem, is_backups_mem, llm_justs, llm_unfiltered, prompts):
    data = {
        "llm_outputs_mem": llm_outputs_mem,
        "is_backups_mem": is_backups_mem,
        "llm_justs": llm_justs,
        "llm_unfiltered": llm_unfiltered,
        "prompts": prompts
    }
    with open('IAG_data.json', 'w') as file:
        json.dump(data, file, indent=4)

def extract_justification_and_action(text):
    # Extracting Justification
    justification = re.search(r'## Justification:(.*?)## Action:', text, re.DOTALL)
    if justification:
        justification_text = justification.group(1).strip()
    else:
        justification_text = None

    # Extracting Action
    action = re.search(r'## Action:\s*(\d+)', text)
    if action:
        action_value = int(action.group(1))
    else:
        action_value = None

    return justification_text, action_value

def extract_justification_and_action_from_json(json_input):
    try:
        import json
        data = json.loads(json_input)
        justification_text = data.get("justification")
        action_value = data.get("action")
        return justification_text, action_value
    except Exception as e:
        print("Error:", e)
        return None, None

def prompt_classification(history, current, back_up_response):
    sys = None
    prompt = incontext_template
    prompt += f"\n## History attacks:\n{history}"
    prompt += f"## Current attack:\n {current}\n\n"
    # prompt += "## Justification: \n{Your justification}\n\n"
    # prompt += "## Action: {your action (0 or 1)}"
    prompt += "```json"
    # print(prompt)
    # return 
    
    output = prompt_open_ai(client, model, prompt, None)
    unfiltered_response = output
    justification, response = extract_justification_and_action_from_json(output)
    extract_justification_and_action(output.strip())
    if(response is None):
        response = back_up_response
        justification = ''
        is_backup = True
    else:
        is_backup = False
        
    return prompt, justification, response, is_backup, unfiltered_response

def sim(x, y):
    return (1-np.abs(x-y)).sum()

def main():
    df = pd.read_csv('/common/home/users/s/shashankc/code/Phishing/IAG/2022-MURIBookChapter-FullData-IAG.csv')
    df = df[['TargetNum', 'Best_Location', 'Best_Payment', 'Best_Penalty', 'Best_Mprob', 'Warning', 'Covered', 'Outcome', 'Action', 'MturkID', 'Condition', 'Block', 'Trial']]
    df['id'] = df.apply(lambda x: x['MturkID'] + '_' + x['Condition'], axis=1)
    df = df.sort_values(by=['Block','Trial'])

    
    #Store Memories
    memories = {}
    for id in df['id']:
        memories[id] = Memory(None, None, None)

    histories = []
    currents = []
    bad_histories = []
    back_up_responses = []
    i = 0
    set = []
    for index, row in df.iterrows():
        id = row['id']
        data = {}
        data['action'] = row['Action']
        data['feedback'] = row['Outcome']
        data['reward'] = row['Best_Payment']
        data['penalty'] = row['Best_Penalty']
        data['mprob'] = row['Best_Mprob']
        data['targetnum'] = row['TargetNum']
        data['location'] = row['Best_Location']
        data['warning'] = row['Warning']
        embedding = np.array([data['reward'], data['penalty'], data['mprob']*10])

        current = f"- Features (TargetNum, Location, Payment, Penalty, Mprob, Warning): ({data['targetnum']}, {data['location']}, {data['reward']}, {data['penalty']}, {data['mprob']}, {data['warning']})"
        currents.append(current)
        #Retrieval
        ret_nodes, _, bad_nodes, _  = memories[id].retrieve(query=data, t=i, n=15, query_embedding=embedding, alpha_t=0.0, ret_least_score=True, rel_func=sim)
        
        #Store History
        j = 1
        concat_history = ''
        for node in ret_nodes:
            concat_history += f'# attack: {j}\n'
            concat_history += f"- Features (TargetNum, Location, Payment, Penalty, Mprob, Warning): ({node.data['targetnum']}, {node.data['location']}, {node.data['reward']}, {node.data['penalty']}, {node.data['mprob']}, {node.data['warning']})\n"
            concat_history += f"- Action: {node.data['action']}\n"
            concat_history += f"- Feedback: {node.data['feedback']}\n\n"
            j += 1
        histories.append(concat_history)

        concat_bad = ''
        j = 1
        for node in bad_nodes:
            concat_bad += f'# attack: {j}\n'
            concat_bad += f"- Features (TargetNum, Location, Payment, Penalty, Mprob, Warning): ({node.data['targetnum']}, {node.data['location']}, {node.data['reward']}, {node.data['penalty']}, {node.data['mprob']}, {node.data['warning']})\n"
            concat_bad += f"- Action: {node.data['action']}\n"
            concat_bad += f"- Feedback: {node.data['feedback']}\n\n"
            j += 1

        bad_histories.append(concat_bad)

        #Store Current Email
        memories[id].insert(None, i, data, embedding)
        i+=1

    df['current'] = currents
    df['history'] = histories

    #Create test set
    df_test = df[(df['Block'] == 3) | (df['Block'] == 4)]

    fpath = '/common/home/users/s/shashankc/code/Phishing/LLM_Prompting/IAG_data.json'
    if(os.path.exists(fpath)):
        data = read_data_from_json(fpath)
        llm_outputs_mem = data["llm_outputs_mem"]
        is_backups_mem = data["is_backups_mem"]
        llm_justs = data["llm_justs"]
        llm_unfiltered = data["llm_unfiltered"]
        prompts = data["prompts"]
        print('Previous Responses Loaded from Disk')
        print(f'Continue Prompting from: {len(prompts)}/{len(df_test)}')

    else:
        llm_outputs_mem = []
        is_backups_mem = []
        llm_justs = []
        llm_unfiltered = []
        prompts = []

    save_interval = 100  # Define the save interval
    print_interval = 100   # Define the print progress interval
    for i in range(len(llm_outputs_mem), len(df_test)):
        row = df_test.iloc[i]
        data = {}
        start = time.time()
        
        if(i % save_interval == 0 and i != 0):  # Save lists periodically
            save_lists(llm_outputs_mem, is_backups_mem, llm_justs, llm_unfiltered, prompts)
            print("Lists saved.")

        current = row['current']
        history = row['history']
        user_action = row['Action']
        
        prompt, llm_just, llm_output_mem, is_backup_mem, llm_unf = prompt_classification(history, current, None)
        is_backups_mem.append(is_backup_mem)
        llm_outputs_mem.append(llm_output_mem)
        llm_justs.append(llm_just)
        llm_unfiltered.append(llm_unf)
        prompts.append(prompt)
        
        end = time.time()
        
        if i % print_interval == 0:
            print(f'Progress: {i}/{len(df_test)}')
            print(f'Time Taken: {end-start}', flush=True)

    # Save the remaining lists
    save_lists(llm_outputs_mem, is_backups_mem, llm_justs, llm_unfiltered, prompts)
    print("Final lists saved.")

    df_test['llm_outputs_mem'] = llm_outputs_mem
    df_test['is_backups_mem'] = is_backups_mem
    df_test['justification'] = llm_justs
    df_test.to_csv(f'{model}_iag.csv', index=False)

    #Number of Backups
    print('Number of Backups:', df_test['is_backups_mem'].sum() / len(df_test))
        

if __name__ == "__main__":
    main()



    

    

    
