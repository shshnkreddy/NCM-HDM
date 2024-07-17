import re
import torch
import openai

def convert_to_llama_format(messages):
    '''Function to convert a message or an array of user and System prompts to llama format.'''
    if isinstance(messages, list) == False:
        messages = f'[INST] {messages} [/INST]'
    return(messages)

def chat_template(content, sys, tokenizer):
    chat = []
    if(sys is not None):
        chat.append({
            'role': 'system',
            'content': f'{sys}'
        })
    chat.append(
        {
            'role': 'user',
            'content': f'{content}'
        }
    )
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def remove_instruction(text):
    # Define a regular expression pattern to find all occurrences of [INST]...[/INST]
    pattern = r'\[INST\].*?\[/INST\]'
    
    # Use re.findall to get all matches
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    # Remove each match from the text
    cleaned_text = text
    for match in matches:
        cleaned_text = cleaned_text.replace(match, '')

    return cleaned_text.strip()

@torch.inference_mode
def prompt_llama_hf(llm, tokenizer, prompts, sys=None):
    # prompts = [convert_to_llama_format(p) for p in prompts]
    prompts = [chat_template(p, sys, tokenizer) for p in prompts]
    inputs = tokenizer(prompts, return_tensors='pt', padding='longest')
    generate_ids = llm.generate(inputs.input_ids, max_length=1024)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    outputs = [remove_instruction(output) for output in outputs]
    return outputs

@torch.inference_mode
def get_embedding_hf(llm, tokenizer, texts, layer='last'):
    inputs = tokenizer(texts, return_tensors='pt', padding='longest')
    embeddings = llm(inputs.input_ids, inputs.attention_mask, output_hidden_states=True)['hidden_states']
    if(layer=='first'):
        embeddings = embeddings[0].mean(dim=1)
    else:
        embeddings = embeddings[-1].mean(dim=1)
    return embeddings

def prompt_llama_vllm(llm, sampling_params, tokenizer, prompts, sys=None):
    # prompts = [convert_to_llama_format(p) for p in prompts]
    prompts = [chat_template(p, sys, tokenizer) for p in prompts]
    raw_outputs = llm.generate(prompts, sampling_params)
    outputs = [out.outputs[0].text for out in raw_outputs]
    return outputs

def prompt_open_ai(client, model, prompt, sys=None):
    if(sys is not None):
        messages=[
            {"role": "system", "content": f"{sys}"},
            {"role": "user", "content": f"{prompt}"}
        ]
    else:
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

