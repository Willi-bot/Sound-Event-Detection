import argparse
import random
import tqdm
import os
import yaml

import torch
import numpy as np
import pandas as pd

from evaluation import evaluate
from utils import get_binary_event_labels, get_classes, get_split_data, get_splits
from data_loading import get_dataloader
from feature_extraction import extract_features, sampling_rates
from models import PrelimModel, BasicCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SED")
    parser.add_argument(
        "--dataset",
        help="name of the dataset to use",
        default='TUT',
        choices=['TUT']
    )
    parser.add_argument(
        "--fold",
        default=1,
        choices=[1, 2, 3, 4],
        type=int
    )
    parser.add_argument(
        '--epochs',
        default=50,
        type=int
    )
    parser.add_argument(
        '--learning-rate',
        default=5e-4,
        type=float
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int
    )
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int
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
        '--dropout',
        default=0.,
        type=float
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_dir = 'dataset/' + args.dataset

    # assignment of files to each split
    train_files, dev_files, test_files = get_splits(args.dataset, args.fold)

    classes = get_classes(args.dataset)
    cls2id = {}
    id2cls = {}
    for i, cls in enumerate(classes):
        cls2id[cls] = i
        id2cls[i] = cls
    num_classes = len(classes)

    # numbers we need for feature extraction and model parameters
    win_length = int((sampling_rates[args.dataset] * args.block_length) / 1000.)
    hop_length = int((1 - args.overlap) * win_length)
    num_blocks = int(args.clip_length / args.block_length)

    feature_filepath = dataset_dir + f'/features/features_{args.clip_length}_{args.n_fft}_{args.n_mels}_{hop_length}_{win_length}.npy'
    # check if extracted features already exist
    if os.path.isfile(feature_filepath):
        # load features
        print('loading features from ' + feature_filepath + "...")
        features = np.load(feature_filepath, allow_pickle=True).item()
    else:
        print("extracting features...")
        features = extract_features(args.dataset, args.clip_length, args.n_fft, args.n_mels, hop_length, win_length)
        print("saving features...")
        np.save(feature_filepath, features)

    label_filepath = dataset_dir + f'/labels/labels_{args.clip_length}_{args.block_length}.npy'
    # check if extracted labels already exist
    if os.path.isfile(label_filepath):
        # load features
        print('loading labels from ' + label_filepath + "...")
        labels = np.load(label_filepath, allow_pickle=True).item()
    else:
        print("extracting labels...")
        labels = get_binary_event_labels(dataset_dir, args.clip_length, args.block_length, cls2id)
        print("saving labels...")
        np.save(label_filepath, labels)

    train_features, train_labels = get_split_data(train_files, features, labels)
    dev_features, dev_labels = get_split_data(dev_files, features, labels)
    test_features, test_labels = get_split_data(test_files, features, labels)

    train_loader = get_dataloader(train_features, train_labels, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = get_dataloader(dev_features, dev_labels, batch_size=args.batch_size, shuffle=False, drop_last=False)
    dev_loader = get_dataloader(test_features, test_labels, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # initialize model & training stuff (optimizer, scheduler, loss func...)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = BasicCNN(num_classes, args.n_mels, gru_hidden_channels=64, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # init rest (like summary writer) TODO
    max_f1 = 0
    best_epoch = 0
    best_results = {}
    best_state = None

    # training
    for epoch in range(args.epochs):
        model.to(device)
        model.train()
        for index, (data) in tqdm.tqdm(
                enumerate(train_loader),
                desc=f"Epoch {epoch}",
                total=len(train_loader)
        ):
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        print(f"dev results at epoch {epoch + 1}: ")
        f1, recall, precision = evaluate(
            model,
            device,
            dev_loader
        )
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"F1-Score: {f1}")

        # show and save results
        if f1 > max_f1:
            max_f1 = f1
            best_epoch = epoch
            best_state = model.cpu().state_dict()
            best_results['f1'] = f1.tolist()
            best_results['recall'] = recall.tolist()
            best_results['precision'] = precision.tolist()

        if epoch - best_epoch > 10:
            print("No improvements for more than 10 epochs. Stopping here...")
            break

    result_dir = "./" + args.dataset + "_results"
    if not os.path.exists(result_dir):
        os.makedirs("./" + args.dataset + "_results")


    print(f"Best dev results found at epoch {best_epoch + 1}:\n{yaml.dump(best_results)}")
    best_results["Epoch"] = best_epoch + 1
    with open(os.path.join(result_dir, f"dev_fold{args.fold}.yaml"), "w") as f:
        yaml.dump(best_results, f)

    torch.save(best_state, os.path.join(
            result_dir, "state.pth.tar"))

    model.load_state_dict(best_state)
    model.eval()
    f1, precision, recall = evaluate(
        model,
        device,
        test_loader
    )
    test_results = {'f1': f1.tolist(), 'recall': recall.tolist(), 'precision': precision.tolist()}
    print(f"Best test results:\n{yaml.dump(test_results)}")
    with open(os.path.join(result_dir, f"test_fold{args.fold}.yaml"), "w") as fp:
        yaml.dump(test_results, fp)
