import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from util.utils import initialize_config, count_parameters
from torchtext.legacy.data import BucketIterator
from data.multi30k import Multi30kData
import numpy as np
import os
import random
import math
import time

#from trainer.base_trainer import  train, evaluate
from util.utils import count_parameters, init_weights, epoch_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config, resume):
    torch.backends.cudnn.deterministic = config["cudnn_deterministic"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    #
    # Data preparation
    #
    #MData = Multi30kData(include_lengths = config['dataset']['include_lengths'])
    MData = initialize_config(config["dataset"])
    train_data, valid_data, test_data, SRC, TRG = MData.splits()

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    #if config["seq2seq_model"]["module"] == "model.seq2seq_with_attention":
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = config['trainer']['batch_size'],
        sort_within_batch = config['data']['sort_within_batch'],
        sort_key = lambda x : len(x.src),
        device = device)
    '''
    else:
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = BATCH_SIZE,
            device = device)
    '''
    if config["seq2seq_model"]["module"] == "model.seq2seq_with_attention" or config["seq2seq_model"]["module"] == "model.seq2seq_with_attention_padded":
        att = initialize_config(config["model_attention"])
        config["model_decoder"]["args"]["attention"] = att

    config["model_encoder"]["args"]["input_dim"] = len(SRC.vocab)
    config["model_decoder"]["args"]["output_dim"] = len(TRG.vocab)

    enc = initialize_config(config["model_encoder"])
    dec = initialize_config(config["model_decoder"])

    config["seq2seq_model"]["args"] = {
        "encoder" : enc,
        "decoder" : dec,
        "device" : device
    }

    if config["seq2seq_model"]["module"] == "model.seq2seq_with_attention_padded":
        config["seq2seq_model"]["args"]["src_pad_idx"] = SRC.vocab.stoi[SRC.pad_token]


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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    #criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    config["loss_function"]["args"]["ignore_index"] = TRG_PAD_IDX
    criterion = initialize_config(config["loss_function"])

    # Training
    N_EPOCHS = config['trainer']['epochs']
    CLIP = config['trainer']['clip']

    best_valid_loss = float('inf')

    os.makedirs(config["checkpoints_dir"], exist_ok=True)

    #
    # Trainer
    #
    trainer_class = initialize_config(config["trainer"], pass_args=False)
    trainer = trainer_class(
        resume=resume,
        model=model,
        loss_function=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_iterator,
        test_dataloader=valid_iterator,
        epochs=config['trainer']['epochs'],
        save_checkpoint_interval=config['trainer']['save_checkpoint_interval'],
        test_interval=config['trainer']['test']['interval'],
        output_dir=config['output_dir'],
        checkpoints_dir=config['checkpoints_dir'],
        find_max=config['trainer']['test']['find_max']
    )
    trainer.train()

    '''
    # Evaluating
    model.load_state_dict(torch.load('tut1-model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    '''

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

    main(config, resume=args.resume)
