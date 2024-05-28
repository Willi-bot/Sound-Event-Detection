import argparse
import random
import tqdm
import os
import yaml

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from evaluation import evaluate
from utils import get_binary_event_labels, get_classes, get_split_data, get_splits, normalize_features, class_weights, dataset_ratio
from data_loading import get_dataloader
from feature_extraction import extract_mel_features, sampling_rates
from models import PrelimModel, BasicRCNN, AdvancedRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SED")
    parser.add_argument(
        "--dataset",
        help="name of the dataset to use",
        default='TUT',
        choices=['TUT', 'desed_2022']
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
    parser.add_argument(
        '--early-stopping',
        type=int
    )
    parser.add_argument(
        '--normalize',
        action='store_true'
    )
    parser.add_argument(
        '--use-weights',
        action='store_true'
    )
    parser.add_argument(
        '--decision-threshold',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--dataset-location',
        default='dataset/'
    )
    parser.add_argument(
        '--use-specaug',
        action='store_true'
    )
    args = parser.parse_args()

    config = vars(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_dir = args.dataset_location + args.dataset

    # assignment of files to each split
    train_files, dev_files, test_files = get_splits(args.dataset, args.fold, args.dataset_location)

    classes = get_classes(args.dataset, args.dataset_location)
    cls2id, id2cls = {}, {}
    for i, cls in enumerate(classes):
        cls2id[cls], id2cls[i] = i, cls
    num_classes = len(classes)

    # numbers we need for feature extraction and model parameters
    # win_length is block length / 1000 (ms -> s) times sampling rate
    win_length = int((sampling_rates[args.dataset] * args.block_length) / 1000.)
    # hop_length depends on overlap and win_length, next frame starts when overlap begins, so 1 - args.overlap
    hop_length = int((1 - args.overlap) * win_length)
    num_blocks = int(args.clip_length / args.block_length)
    # round up, because we keep last frame that might not be complete
    num_features_frames = np.ceil(float(win_length) / hop_length)

    feature_filepath = dataset_dir + f'/features/features_{args.clip_length}_{args.n_fft}_{args.n_mels}_{hop_length}_{win_length}.npy'
    mfcc_feature_filepath = dataset_dir + f'/features/mfcc_features_{args.clip_length}_{args.n_fft}_{args.n_mels}_{hop_length}_{win_length}.npy'
    # check if extracted features already exist
    if os.path.isfile(feature_filepath):
        # load features
        print('loading features from ' + feature_filepath + "...")
        features = np.load(feature_filepath, allow_pickle=True).item()
    else:
        features = None

    if features is None:
        print("extracting features...")
        features = extract_mel_features(args.dataset, args.clip_length, args.n_fft, args.n_mels, hop_length, win_length)
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
        labels = get_binary_event_labels(args.dataset, args.clip_length, args.block_length, cls2id, args.dataset_location)
        print("saving labels...")
        np.save(label_filepath, labels)

    if args.normalize:
        features = normalize_features(features)

    train_features, train_labels = get_split_data(train_files, features, labels, dataset=args.dataset)
    dev_features, dev_labels = get_split_data(dev_files, features, labels, dataset=args.dataset)
    test_features, test_labels = get_split_data(test_files, features, labels, dataset=args.dataset)

    train_loader = get_dataloader(train_features, train_labels, batch_size=args.batch_size, shuffle=True, drop_last=True, use_specaug=args.use_specaug)
    test_loader = get_dataloader(dev_features, dev_labels, batch_size=args.batch_size, shuffle=False, drop_last=False, use_specaug=False)
    dev_loader = get_dataloader(test_features, test_labels, batch_size=args.batch_size, shuffle=False, drop_last=False, use_specaug=False)

    # initialize model & training stuff (optimizer, scheduler, loss func...)
    if args.use_weights:
        weights = class_weights[args.dataset]
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(weights).to(device), pos_weight=dataset_ratio[args.dataset])
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=dataset_ratio[args.dataset])
    model = AdvancedRCNN(num_classes, args.dropout)
    config['model'] = {'name': 'AdvancedCRNN'}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    config['optimizer'] = 'Adam'

    result_dir = "./" + args.dataset + "_results"
    if not os.path.exists(result_dir):
        os.makedirs("./" + args.dataset + "_results")
    with open(os.path.join(result_dir, f"config.yaml"), "w") as fp:
        yaml.dump(config, fp)


    # init rest
    max_f1 = 0
    best_epoch = 0
    best_results = {}
    best_class_results = {}
    best_state = None

    writer = SummaryWriter(log_dir="./" + args.dataset + "_results")

    # training
    for epoch in range(1, args.epochs + 1):
        model.to(device)
        model.train()
        for index, (data) in tqdm.tqdm(
                enumerate(train_loader),
                desc=f"Epoch {epoch}",
                total=len(train_loader)
        ):
            features, labels = data
            features = features.to(device)
            output = model(features).view(-1, num_classes)
            labels = labels.view(-1, num_classes).to(device)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        dev_results, dev_class_results = evaluate(
            model,
            device,
            dev_loader,
            id2cls
        )
        print(f"dev results at epoch {epoch}:\n{yaml.dump(dev_results)}")
        writer.add_scalar('F1-Score/dev', dev_results['f1'], epoch)
        for cls in cls2id.keys():
            writer.add_scalar(f'F1-Score_({cls})', dev_class_results[cls]['f1'], epoch)

        # show and save results
        if dev_results['f1'] > max_f1:
            max_f1 = dev_results['f1']
            best_epoch = epoch
            best_state = model.cpu().state_dict()
            best_results = dev_results.copy()
            best_class_results = dev_class_results.copy()

        if args.early_stopping is not None and epoch - best_epoch > args.early_stopping:
            print(f"No improvements for more than {args.early_stopping} epochs. Stopping here...")
            break

    print(f"Best dev results found at epoch {best_epoch + 1}:\n{yaml.dump(best_results)}")
    best_results["Epoch"] = best_epoch + 1
    with open(os.path.join(result_dir, f"dev_fold{args.fold}.yaml"), "w") as f:
        yaml.dump(best_results, f)
        yaml.dump(best_class_results, f)

    torch.save(best_state, os.path.join(
            result_dir, "state.pth.tar"))

    model.load_state_dict(best_state)
    model.eval()
    test_results, test_class_results = evaluate(
        model,
        device,
        test_loader,
        id2cls
    )
    print(f"Best test results:\n{yaml.dump(test_results)}")
    with open(os.path.join(result_dir, f"test_fold{args.fold}.yaml"), "w") as fp:
        yaml.dump(test_results, fp)
        yaml.dump(test_class_results, fp)
