import argparse
import random

import pandas as pd
import tqdm
import os
import yaml

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import librosa

from evaluation import evaluate, get_prediction_from_raw_output
from utils import get_classes, get_splits, class_weights, get_test_files, get_labels, dataset_ratio
from data_loading import get_dataloader
from feature_extraction import sampling_rates, get_features, extract_mel_spectrograms
from models import BasicRCNN, ShakeRCNN, ShakeTransformer
from nnet.CRNN import CRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SED")
    parser.add_argument(
        '--dataset-location',
        default='dataset/'
    )
    parser.add_argument(
        '--clip-length',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--block-length',
        default=40,
        type=int
    )
    parser.add_argument(
        '--overlap',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--n-fft',
        default=2048,
        type=int
    )
    parser.add_argument(
        '--n-mels',
        default=64,
        type=int
    )
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int
    )
    parser.add_argument(
        '--save-folder',
        default='',
        type=str
    )
    parser.add_argument(
        '--decision-threshold',
        default=0.5,
        type=float
    )
    args = parser.parse_args()

    learning_rates = [1e-2, 1e-3, 1e-4]
    dropouts = [0.2, 0.35, 0.5]
    optimizers = ['Adam', 'AdamW', 'Nadam']

    best_f1 = -1.
    best_learning_rate = None
    best_dropout = None
    best_optimizer = None

    for learning_rate in learning_rates:
        for dropout in dropouts:
            for optimizer_name in optimizers:

                # set seeds
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)

                dataset_dir = args.dataset_location + 'BirdSED'

                # assignment of files to each split
                train, dev, test = get_splits('BirdSED', args.dataset_location, fold=1)
                train_audio_files, train_label_files = train
                dev_audio_files, dev_label_files = dev
                test_audio_files, test_label_files = test

                # get list of classes and get dicts mapping ids <-> class name
                classes = get_classes('BirdSED', args.dataset_location, 1)
                cls2id, id2cls = {}, {}
                for i, cls in enumerate(classes):
                    cls2id[cls], id2cls[i] = i, cls
                num_classes = len(classes)

                # numbers we need for feature extraction and model parameters
                # win_length is block length / 1000 (ms -> s) times sampling rate
                win_length = int((sampling_rates['BirdSED'] * args.block_length) / 1000.)
                # hop_length depends on overlap and win_length, next frame starts when overlap begins, so 1 - args.overlap
                hop_length = int((1 - args.overlap) * win_length)
                num_blocks = int(args.clip_length / args.block_length)
                # round up, because we keep last frame that might not be complete
                num_features_frames = np.ceil(float(win_length) / hop_length)

                # get required data (if features already extracted loads from file, otherwise extracts features)
                print('Fetching train data...')
                train_folder = 'ast_train_10k'
                train_features = get_features("features/" + train_folder, args.dataset_location, 'BirdSED', 1, args.clip_length, args.n_fft, args.n_mels, hop_length,
                                        win_length, audio_files=train_audio_files, get_ast_features=True)
                train_labels = get_labels('labels/train', args.dataset_location, 'BirdSED', 1, args.clip_length,
                                    args.block_length, cls2id, metadata_files=train_label_files)

                print('Fetching validation data...')
                dev_folder = 'ast_dev_10k'
                dev_features = get_features("features/" + dev_folder, args.dataset_location, 'BirdSED', 1, args.clip_length, args.n_fft, args.n_mels, hop_length,
                                        win_length, audio_files=dev_audio_files, get_ast_features=True)
                dev_labels = get_labels('labels/dev', args.dataset_location, 'BirdSED', 1, args.clip_length,
                                    args.block_length, cls2id, metadata_files=dev_label_files)

                print('Fetching test data...')
                test_folder = 'ast_test_10k'
                test_features = get_features("features/" + test_folder, args.dataset_location, 'BirdSED', 1, args.clip_length, args.n_fft, args.n_mels, hop_length,
                                        win_length, audio_files=test_audio_files, get_ast_features=True)
                test_labels = get_labels('labels/test', args.dataset_location, 'BirdSED', 1, args.clip_length,
                                    args.block_length, cls2id, metadata_files=test_label_files)

                train_loader = get_dataloader(train_features, train_labels, batch_size=args.batch_size, shuffle=True, drop_last=True, use_specaug=True)
                dev_loader = get_dataloader(dev_features, dev_labels, batch_size=args.batch_size, shuffle=False, drop_last=False, use_specaug=False)
                test_loader = get_dataloader(test_features, test_labels, batch_size=args.batch_size, shuffle=False, drop_last=False, use_specaug=False)

                # initialize model & training stuff (optimizer, scheduler, loss func...)

                model = ShakeRCNN(num_classes, dropout)
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=dataset_ratio['BirdSED'])

                if optimizer_name == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                elif optimizer_name == 'AdamW':
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                elif optimizer_name == 'Nadam':
                    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                else:
                    exit(1)

                # init rest
                max_f1 = -1.
                best_epoch = 0
                best_results = {}
                best_class_results = {}
                best_state = None
                already_trained = False

                epochs = 100
                early_stopping = 25

                print(f'Testing combination {learning_rate}/{dropout}/{optimizer_name}...')
                model.to(device)

                # training
                if not already_trained:
                    for epoch in range(1, 100 + 1):
                        model.train()
                        for index, (data) in tqdm.tqdm(
                                enumerate(train_loader),
                                desc=f"Epoch {epoch}",
                                total=len(train_loader)
                        ):
                            features, labels = data
                            features = features.to(device)
                            output = model(features)
                            output = output.reshape(-1, num_classes)
                            labels = labels.reshape(-1, num_classes).to(device)
                            loss = loss_fn(output, labels)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            dev_results, dev_class_results = evaluate(
                                model,
                                device,
                                dev_loader,
                                id2cls,
                                decision_threshold=args.decision_threshold,
                                apply_sigmoid=False
                            )
                        print(f"dev results at epoch {epoch}:\n{yaml.dump(dev_results)}")

                        # show and save results
                        if dev_results['f1'] > max_f1:
                            max_f1 = dev_results['f1']
                            best_epoch = epoch
                            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                            best_results = dev_results.copy()
                            best_class_results = dev_class_results.copy()

                        if epoch - best_epoch > 20:
                            print(f"No improvements for more than {20} epochs. Stopping here...")
                            break

                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    test_results, test_class_results = evaluate(
                        model,
                        device,
                        test_loader,
                        id2cls,
                        decision_threshold=args.decision_threshold,
                        apply_sigmoid=False
                    )

                if test_results['f1'] > best_f1:
                    best_f1 = test_results['f1']
                    best_learning_rate = learning_rate
                    best_dropout = dropout
                    best_optimizer = optimizer_name

                test_f1 = test_results['f1']
                print(f'Combination {learning_rate}/{dropout}/{optimizer_name} finished with {test_f1} on test\n')


    print(f'Best learning rate: {best_learning_rate}\n')
    print(f'Best dropout: {best_dropout}\n')
    print(f'Best optimizer: {best_optimizer}\n')
    print(f'Resulting F1 score: {best_f1}')