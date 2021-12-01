'''
Train a T5 model for Grammatical Error Correction
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, T5ForConditionalGeneration
from data_prep import DataTensorLoader
import sys
import os
import argparse
from tools import AverageMeter, get_default_device

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(train_loader, model, output_to_loss, optimizer, epoch, device, print_freq=5):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (inp_id, mask, out_id) in enumerate(train_loader):

        inp_id = inp_id.to(device)
        mask = mask.to(device)
        out_id = out_id.to(device)

        # Forward pass
        outputs = model(input_ids=inp_id, attention_mask=mask, labels=out_id)
        loss = output_to_loss(outputs)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), inp_id.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), loss=losses))

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('INC', type=str, help='Path to incorrect training data')
    commandLineParser.add_argument('CORR', type=str, help='Path to correct training data')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=2, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.000001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=10, help="Specify scheduler rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    args = commandLineParser.parse_args()

    set_seeds(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data as tensors
    dataloader = DataTensorLoader()
    input_ids, input_mask, output_ids = dataloader.get_train(args.INC, args.CORR)

    # Use dataloader to handle batches
    train_ds = TensorDataset(input_ids, input_mask, output_ids)
    train_dl = DataLoader(train_ds, batch_size=args.B, shuffle=True)

    # Initialise model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.sch])

    # loss from output - criterion is wrapped inside model
    loss_from_output = lambda a: a[0]

    # Train
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, loss_from_output, optimizer, epoch, device)
        scheduler.step()

    # Save the trained model
    state = model.state_dict()
    torch.save(state, args.OUT)