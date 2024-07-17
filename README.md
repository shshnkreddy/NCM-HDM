## Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans

Authors: Changyu Chen*, Shashank Reddy Chirra*, Maria José Ferreira*, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

*Equal Contribution

This study introduces two attention-based neural network models for cognitive modeling of individual human decision-making in dynamic settings. Tested on datasets from phishing email detection and cybersecurity attack scenarios, these models outperform Instance Based Learning (IBL) and GPT3.5 in representing human decision-making. The research also reveals that all models predict decisions more accurately for humans who perform better at the task. The study also explores decision explanations based on the model's predictive factors, offering promising insights for future applications of neural networks in cognitive modeling of human decision-making.

## Installation 

To install the necessary dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation and Training 

### IL-PIMM

#### Data Preperation:
```
[CHANGYU]
```

#### Training:
```
[CHANGYU]
```

### TL-PIMM

#### Data Preperation:
```
[CHANGYU]
```

#### Training:
To train the simple attention model, run the following command:
```
bash shallow_transformer/run_train.sh
```

### GPT 3.5-Turbo

#### Data Preperation (Phishing Dataset only):
To generate embeddings for emails (for retrieval from the memory), run the following command: 
```
bash gpt/run_generate_embeddings.sh
```

#### Training:
To generate results using the gpt3.5-turbo model, run the following command:
```
bash gpt/run_prompt_gpt_{iag/phishing}.sh
```

### IBL

```
[MARIA]
```

## More information

Paper Link: 

Contact: cychen.2020@phdcs.smu.edu.sg
