import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from util.utils import initialize_config, count_parameters
from torchtext.legacy.data import BucketIterator
from data.multi30k import Multi30kData

import spacy
import numpy as np
import os
import random
import math
import time

from trainer.base_trainer import  train, evaluate
from util.utils import count_parameters, init_weights, epoch_time

torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config):

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    #
    # Data preparation
    #
    MData = Multi30kData()
    train_data, valid_data, test_data, SRC, TRG = MData.splits()

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")


    BATCH_SIZE = config['trainer']['batch_size']

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE,
        device = device)


    config["model_encoder"]["args"]["input_dim"] = len(SRC.vocab)
    config["model_decoder"]["args"]["output_dim"] = len(TRG.vocab)

    enc = initialize_config(config["model_encoder"])
    dec = initialize_config(config["model_decoder"])

    config["seq2seq_model"]["args"] = {
        "encoder" : enc,
        "decoder" : dec,
        "device" : device
    }
    
    model = initialize_config(config["seq2seq_model"])
    model = model.to(device)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    #
    # Optimizer
    #
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        weight_decay=config["optimizer"]["weight_decay"]
    )

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    #criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    config["loss_function"]["args"]["ignore_index"] = TRG_PAD_IDX
    criterion = initialize_config(config["loss_function"])

    # Training
    N_EPOCHS = config['trainer']['epochs']
    CLIP = 1

    best_valid_loss = float('inf')

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            print(f'Saving checkpoint! Best Loss : {valid_loss}| Old Loss: {best_valid_loss} ')
            best_valid_loss = valid_loss 
            torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], 'best_model'))

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


    # Evaluating
    model.load_state_dict(torch.load('tut1-model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CRN")
    parser.add_argument("-c", "--config", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-p", "--preloaded_model_path", type=str, help="Path of the *.Pth file of the model.")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert not args.resume, "Resume conflict with preloaded model. Please use one of them."

    with open(args.config, "r") as f:
        config = json.load(f)

    main(config)