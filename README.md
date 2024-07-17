## Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans

Authors: Changyu Chen*, Shashank Reddy Chirra*, Maria José Ferreira*, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

*Equal Contribution

## Installation 

To install the necessary dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation and Training 

### IL-PIMM

#### Data Preperation:
'''
[CHANGYU]
'''

#### Training:
'''
[CHANGYU]
'''

### TL-PIMM

#### Data Preperation:
'''
[CHANGYU]
'''

#### Training:
To train the simple attention model, run the following command:
'''
bash shallow_transformer/run_train.sh
'''

### GPT 3.5-Turbo

#### Data Preperation (Phishing Dataset only):
To generate embeddings for emails (for retrieval from the memory), run the following command: 
'''
bash gpt/run_generate_embeddings.sh
'''

#### Training:
To generate results using the gpt3.5-turbo model, run the following command:
'''
bash gpt/run_prompt_gpt_{iag/phishing}.sh
'''

### IBL

'''
[MARIA]
'''

