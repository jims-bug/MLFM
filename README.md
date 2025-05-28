# Aspect-Based Sentiment Analysis with Semantic and Syntactic Enhanced Multi-Layer Fusion Model
Code and datasets of our paper "Aspect-Based Sentiment Analysis with Semantic and Syntactic Enhanced Fusion Network"


## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==3.2.0
- cython==0.29.13
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.

## Preparation

1. Download Bert(`bert-base-uncased`) from [(https://huggingface.co/bert-base-uncased)](https://huggingface.co/bert-base-uncased) and put it into  `MLFM/bert-base-uncased` directory.


2. Prepare vocabulary with:

   `sh build_vocab.sh`

## Training

To train the MLFM model, run:

`sh run.sh`

## Credits

The code and datasets in this repository are based on [SSEGCN_ABSA](https://github.com/zhangzheng1997/SSEGCN-ABSA.git) .

