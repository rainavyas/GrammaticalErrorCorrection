'''
Calulate Performance Metrics of GEC system from prediction files
'''

import sys
import os
import argparse
import string
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu

def get_sentences_dict(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    exclude = set(string.punctuation)
    id2text = {}
    for l in lines:
        parts = l.split()
        id = parts[0]
        text = ' '.join(parts)
        # remove punctuation
        text = ''.join(ch for ch in text if ch not in exclude)
        id2text[id] = text
    return id2text

def align_data(pred_dict, corr_dict):
    pred_sens = []
    corr_sens = []
    for i, (id, text) in enumerate(corr_dict.items()):
        try:
            pred_sens.append(pred_dict[id])
            corr_sens.append(text)
        except:
            # print(f'{i}) {id} in corrected but not in predicted')
            pass
    return pred_sens, corr_sens

def batch_bleu_score(pred_sens, corr_sens):
    total_score = 0
    for pred, corr in zip(pred_sens, corr_sens):
        candidate = pred.split()
        reference = [corr.split()]
        score = sentence_bleu(reference, candidate)
        total_score += score
    return total_score/len(pred_sens)

def batch_gleu_score(pred_sens, corr_sens):
    total_score = 0
    for pred, corr in zip(pred_sens, corr_sens):
        candidate = pred.split()
        reference = [corr.split()]
        score = sentence_gleu(reference, candidate)
        total_score += score
    return total_score/len(pred_sens)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('PRED', type=str, help='Path to output predicted data')
    commandLineParser.add_argument('CORR', type=str, help='Path to correct output test data')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/calculate_scores.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Get sentences and align
    pred_id2text = get_sentences_dict(args.PRED)
    corr_id2text = get_sentences_dict(args.CORR)
    pred_sens, corr_sens = align_data(pred_id2text, corr_id2text)

    # Get BLEU score
    bleu = batch_bleu_score(pred_sens, corr_sens)
    print(f'BLEU Score: {bleu}')

    # Get GLEU score
    gleu = batch_gleu_score(pred_sens, corr_sens)
    print(f'GLEU Score: {gleu}')