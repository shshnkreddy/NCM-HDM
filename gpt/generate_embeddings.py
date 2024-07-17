from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import json
import fire
import numpy as np

def filter_history(text):    
    cleaned_text = re.sub(r'## History Emails and Responses:.*?## Current Email:\n', 'Email:', text, flags=re.DOTALL)
    return cleaned_text

def main(file_path):
    model = SentenceTransformer('sentence-transformers/gtr-t5-xxl')
    # Initialize an empty list to store dictionaries
    data = []

    # Read the file line by line and load each line as a JSON object
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming each line is a valid JSON string representing a dictionary
            dictionary = json.loads(line)

            # Append the dictionary to the list
            data.append(dictionary)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    df['current'] = df['input'].apply(lambda x: filter_history(x))
    df['current'] = df['current'].apply(lambda x: x.strip())

    # print(df.columns)
    for column in df.columns:
        df[column] = df[column].apply(lambda x: str(x))
        df[column] = df[column].astype('str')

    def get_embeddings(texts, batch_size=4096, layer='last'):
        embeddings = []
        for i in range(0, len(df), batch_size):
            l = i
            r = min(i+batch_size, len(df))
            batch_text = texts[l:r]
            # _embeddings = get_embedding_hf(llm, tokenizer, batch_text, layer='last').detach().cpu().numpy()
            # _embeddings = np.random.rand((r-l, 768))
            _embeddings = model.encode(batch_text)
            embeddings.append(_embeddings)
            print(f'Step: {i/len(df)*100}')
        return np.vstack(embeddings)

    embeddings = get_embeddings(df['current'].values.tolist(), batch_size=4, layer='last').tolist()
    df['current_embedding'] = embeddings

    print(df.info())

if __name__ == '__main__':
    fire.Fire(main)