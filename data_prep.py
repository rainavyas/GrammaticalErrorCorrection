'''
Prepare the CLC FCE-public dataset as tokenized torch
(input and output) ids tensor and attention mask
'''

import torch
import torch.nn as nn
from transformers import T5Tokenizer


class DataTensorLoader():
    def __init__(self, max_len=400):
        self.max_len = max_len
    
    def _get_sentences(self, data_path):

        with open(data_path, 'r') as f:
            lines = f.readlines()
        lines = [l.rstrip('\n') for l in lines]
        id2text = {}
        for l in lines:
            parts = l.split()
            id = parts[0]
            text = ' '.join(parts)
            id2text[id] = text
        return id2text

    def _get_data(self, original_data_path, corrected_data_path):

        original_id2text = self._get_sentences(original_data_path)
        corrected_id2text = self._get_sentences(corrected_data_path)

        original_sentences = []
        corrected_sentences = []
        for i, (id, text) in enumerate(corrected_id2text.items()):
            try:
                original_sentences.append(original_id2text[id])
                corrected_sentences.append(text)
            except:
                print(f'{i}) {id} in corrected but not in original')
        assert len(original_sentences) == len(corrected_sentences), "Input and Output samples misaligned"

        # prep input tensors - original
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        encoded_inputs = tokenizer(original_sentences, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = encoded_inputs['input_ids']
        input_mask = encoded_inputs['attention_mask']

        # prep output tensors - corrected -> use '-100' for masked positions
        encoded_inputs = tokenizer(corrected_sentences, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        output_ids = encoded_inputs['input_ids']
        mask = encoded_inputs['attention_mask']
        output_ids[mask==0] = -100

        return input_ids, input_mask, output_ids


    def get_train(self, original_data_path, corrected_data_path):
        return self._get_data(original_data_path, corrected_data_path)

    def get_test(self, original_data_path, corrected_data_path):
        return self._get_data(original_data_path, corrected_data_path)
