'''
Inference of GEC model

i.e. generate and save predicted sentence outputs from GEC model
'''
from data_prep import DataTensorLoader
import sys
import os
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify trained model th file')
    commandLineParser.add_argument('INC', type=str, help='Path to input test data')
    commandLineParser.add_argument('CORR', type=str, help='Path to correct output test data')
    commandLineParser.add_argument('OUT', type=str, help='Path to output prediction file')
    commandLineParser.add_argument('--num_beams', type=int, default=1, help='Number of beams to use')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the data as tensors
    dataloader = DataTensorLoader()
    input_ids, input_mask, _, identifiers = dataloader.get_test(args.INC, args.CORR)
    # #temp
    # samples = 20
    # input_ids = input_ids[:samples]
    # input_mask = input_mask[:samples]
    # identifiers = identifiers[:samples]
    print("Loaded data")

    # Load trained model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Decode ids into sentences
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    prediction_sentences = []
    for i, (inp_id, mask) in enumerate(zip(input_ids, input_mask)):
        # Generate prediction ids
        prediction_ids = model.generate(
        input_ids = inp_id.unsqueeze(dim=0),
        attention_mask = mask.unsqueeze(dim=0),
        num_beams = args.num_beams,
        do_sample = False,
        max_length = 60,
        length_penalty = 1.0,
        early_stopping = True,
        use_cache = True,
        num_return_sequences = 1
    )
        print(f'Decoding {i}/{len(prediction_ids)}')
        prediction_sentences.append(tokenizer.decode(prediction_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    assert len(prediction_sentences) == len(identifiers), "Number of ids don't match number of predictions"

    # Save predictions
    with open(args.OUT, 'w') as f:
        for id, sentence in zip(identifiers, prediction_sentences):
            f.write(f'{id} {sentence}\n')


