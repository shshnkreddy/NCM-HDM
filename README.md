## Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans

Authors: Changyu Chen*, Shashank Reddy Chirra*, Maria José Ferreira*, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

*Equal Contribution

This study introduces two attention-based neural network models for cognitive modeling of individual human decision-making in dynamic settings. Tested on datasets from phishing email detection and cybersecurity attack scenarios, these models outperform Instance Based Learning (IBL) and GPT3.5 in representing human decision-making. The research also reveals that all models predict decisions more accurately for humans who perform better at the task. The study also explores decision explanations based on the model's predictive factors, offering promising insights for future applications of neural networks in cognitive modeling of human decision-making.

## Setup environment

To install the necessary dependencies, please run the following command:
```bash
conda create -n ncm
pip install -r requirements.txt

conda create -n llm # creat another one for LLM
git clone https://github.com/dvlab-research/LongLoRA.git
# (following instructios of LongLora to install dependencies)
```
Note: use `llm` conda environment for TL-PIMM, while `ncm` for others.

## Data Preparation and Training 
### Data Preperation
Please download the processed data from: 
- Phishing: [dropbox.com](https://www.dropbox.com/scl/fi/ten1liealdf2r33kzsyta/phishing.zip?rlkey=lc84xs11iiu9v23j4s4n9b5ev&dl=0)
- IAG: [dropbox.com](https://www.dropbox.com/scl/fi/1yyeapx0hvanox28a1ddy/IDG.zip?rlkey=eehgs3jfj2maymsa62brsutag&dl=0)

The raw data is released by [DDMLab](https://www.cmu.edu/dietrich/sds/ddmlab/) and can be accessed via [https://osf.io/c7ntu/](https://osf.io/c7ntu/?view_only=0e6261b5e818440495d9917044611758) and https://osf.io/r83ag/. 

For preparing data for experiment of GPT 3.5-Turbo (Phishing Dataset only), run the following command: 
```
python gpt/generate_embeddings.py --filepath=<path to phishing_response.json> --out_dir=<path to store the embedded dataset>
```

`phishing_response.json` can be found in Phishing dataset.

### IL-PIMM
To train the IL-PIMM, run the following command:
```
bash il-pimm/run_train.sh
```

Note: provide argument `root_dir` and `data_dir`. `root_dir` is the directory where you want to save the results; `data_dir` is the directory of your `phishing-Emd` folder (find it in Phishing dataset).

### TL-PIMM
To train the TL-PIMM, run the following command:
```
bash tl-pimm/run_train.sh
```

Note: use `llm` conda environment; provide `data_path` and `output_dir`, `data_path` is the path of the `train.json` file in `phishing-Text` folder. 

### GPT 3.5-Turbo
To generate results using the gpt3.5-turbo model, run the following command:
```
python gpt/python prompt_gpt_{iag/phishing}.py --file_path={path to embedded_dataset/IAG dataset}
```

Note: esnsure that you insert your open-ai api key in the OPEN_AI_KEY field in 'gpt/key.py'.

### IBL
We utilize the implementation of IBL for [phishing](https://github.com/DDM-Lab/PhishingTrainingTask) and [IAG](https://github.com/DDM-Lab/InsiderAttackGame) respectively. 

## More information

Paper Link: 

Contact: cychen.2020@phdcs.smu.edu.sg, shashankc@smu.edu.sg, ddmlab@andrew.cmu.edu
