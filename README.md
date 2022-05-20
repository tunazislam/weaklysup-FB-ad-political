# weaklysup-FB-ad-political

This repository contains code for paper titled "Weakly Supervised Learning for Analyzing Political Campaigns on Facebook", [ICWSM 2023](https://www.icwsm.org/2023/index.html/).

## Data:

1. Please download the 'data' folder from following link [[Data]](https://drive.google.com/drive/u/1/folders/1_t0SiOVmHq4hCPSaAcj51HVcTQPO10Et)
2. Links of 30 news article for each of 13 issues are inside 30_news_articles_urls.zip folder.
3. The dataset is parsed in an usable format for the codes in data.pickle. This file is needed to run the model. This file can be found inside Data folder.



## Computing Machine:

```
Supermicro SuperServer 7046GT-TR

Two Intel Xeon X5690 3.46GHz 6-core processors

48 GB of RAM

Linux operating system

Four NVIDIA GeForce GTX 1060 GPU cards for CUDA/OpenCL programming

```

## Software Packages and libraries:

python 3.6.6

PyTorch 1.1.0

jupiter notebook

pandas

pickle

gensim

nltk

nltk.tag

spacy

emoji

sklearn

statsmodels

scipy

matplotlib

numpy

preprocessor

transformers

## Code: 

### For weakly supervised graph embedding based model:

(1) The main function is implemented in main.py

(2) The embedding learning is implemented in Embedder.py

(3) The Bi-directional LSTM is implemented in BLSTM.py


### Supervised Baseline:

Data for supervised baseline can be found inside /Data/supervised_data/ folder. 

Codes for supervised baseline can be found inside /Code/Baselines/ folder.

(1) For BiLSTM_glove baseline, please run Code/Baselines/baseline_glove_lstm_FB_political.ipynb

(2) For BERT fine-tuned baseline, please run Code/Baselines/baseline_bert.py

### For Analysis:

Run Code/analysis_fb_Ad.ipynb
