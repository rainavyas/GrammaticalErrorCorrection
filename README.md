# GrammaticalErrorCorrection
Train a Transformer-based seq2seq GEC model to correct grammatical errors in sentences.

# Objective

The aim is to peform the NLP sequence-to-sequence task of generating a grammatically correct version of an input sentence. This is achieved in this work using a Trasnformer based encoder-decoder architecture, specifically, a [T5 model](https://huggingface.co/transformers/model_doc/t5.html).

[Cambridge Learner Corpus (CLC) data, FCE split](https://www.comp.nus.edu.sg/~nlp/conll14st.html) is used for training of this model.


# Requirements

python3.7 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers

pip install sentencepiece


# Training

Fork the repository (and clone).

Download the data.

Run the _train.py_ scripts with desired arguments in your terminal. For example,:

_python ./train.py my_trained_model.th input_data_path corrected_data_path --B=8 --lr=0.00001 --epochs=2 --seed=1_

# Using the Errant Tool

This tool used to evaluate the GEC model. The tool is [here](https://github.com/chrisjbryant/errant). The paper is [here](https://aclanthology.org/N12-1067.pdf).

## Get the tool:

`pip install -U pip setuptools wheel`

`pip install errant`

`python -m spacy download en`


## How to use the tool

Assume you have three files with a sentence per line (generate using my align data script):
file1.inc  .... original incorrect sentences
file2.pred ... Model prediction sentences
file3.corr ... True manual labels, reference sentences

There are two stages:

1) Generate the edits files

`errant_parallel -orig file1.inc -cor file2.pred -out edits-pred.m2`

`errant_parallel -orig file1.inc -cor file2.corr -out edits-corr.m2`

2) Calculate the overlap F-score

`errant_compare -hyp edits-pred.m2 -ref edits-corr.m2`


