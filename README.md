# GrammaticalErrorCorrection
Train a Transformer-based seq2seq GEC model to correct grammatical errors in sentences.

# Objective

The aim is to peform the NLP sequence-to-sequence task of generating a grammatically correct version of an input sentence. This is achieved in this work using a Trasnformer based encoder-decoder architecture, specifically, a T5 model: https://huggingface.co/transformers/model_doc/t5.html. CLC data is used for training of this model.


# Requirements

python3.7 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers


# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example,:

_python ./train.py my_model.th --B=8 --lr=0.00001 --epochs=2 --seed=1_
