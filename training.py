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
from models import BasicRCNN, AdvancedRCNN, AttentionRCNN
from nnet.CRNN import CRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SED")
    parser.add_argument(
        "--dataset",
        help="name of the dataset to use",
        default='TUT',
        choices=['TUT', 'desed_2022', 'BirdSED']
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
    parser.add_argument(
        '--model',
        default='Basic',
        choices=['Basic', 'Advanced', 'Baseline', 'Attention'],
        required=True,
        type=str
    )
    args = parser.parse_args()

    config = vars(args).copy()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_dir = args.dataset_location + args.dataset

    # assignment of files to each split
    train, dev, test = get_splits(args.dataset, args.dataset_location, fold=args.fold)
    train_audio_files, train_label_files = train
    dev_audio_files, dev_label_files = dev
    test_audio_files, test_label_files = test

    classes = get_classes(args.dataset, args.dataset_location, args.fold)
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

    train_features = get_features("features/train", args.dataset_location, args.dataset, args.fold, args.clip_length, args.n_fft, args.n_mels, hop_length,
                            win_length, audio_files=train_audio_files)
    train_labels = get_labels('labels/train', args.dataset_location, args.dataset, args.fold, args.clip_length,
                        args.block_length, cls2id, metadata_files=train_label_files)

    dev_features = get_features("features/dev", args.dataset_location, args.dataset, args.fold, args.clip_length, args.n_fft, args.n_mels, hop_length,
                            win_length, audio_files=dev_audio_files)
    dev_labels = get_labels('labels/dev', args.dataset_location, args.dataset, args.fold, args.clip_length,
                        args.block_length, cls2id, metadata_files=dev_label_files)

    test_features = get_features("features/test", args.dataset_location, args.dataset, args.fold, args.clip_length, args.n_fft, args.n_mels, hop_length,
                            win_length, audio_files=test_audio_files)
    test_labels = get_labels('labels/test', args.dataset_location, args.dataset, args.fold, args.clip_length,
                        args.block_length, cls2id, metadata_files=test_label_files)

    train_loader = get_dataloader(train_features, train_labels, batch_size=args.batch_size, shuffle=True, drop_last=True, use_specaug=args.use_specaug)
    dev_loader = get_dataloader(dev_features, dev_labels, batch_size=args.batch_size, shuffle=False, drop_last=False, use_specaug=False)
    test_loader = get_dataloader(test_features, test_labels, batch_size=args.batch_size, shuffle=False, drop_last=False, use_specaug=False)

    # initialize model & training stuff (optimizer, scheduler, loss func...)
    if args.use_weights:
        weights = class_weights[args.dataset]
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(weights).to(device), pos_weight=dataset_ratio[args.dataset])
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=dataset_ratio[args.dataset])

    if args.model == 'Basic':
        model = BasicRCNN(num_classes, 64)
    elif args.model == 'Advanced':
        model = AdvancedRCNN(num_classes, args.dropout)
    elif args.model == 'Baseline':
        model = CRNN(dropout=0.5, rnn_layers=2, n_in_channel=1, nclass=num_classes, attention=True, n_RNN_cell=128,
                     activation='glu', rnn_type='BGRU', kernel_size=[3, 3, 3, 3, 3, 3, 3],
                     padding=[1, 1, 1, 1, 1, 1, 1], stride=[1, 1, 1, 1, 1, 1, 1],
                     nb_filters=[ 16, 32, 64, 128, 128, 128, 128],
                     pooling=[ [ 2, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ] ],
                     dropout_recurrent=0)
        loss_fn = torch.nn.BCELoss()
    elif args.model == 'Attention':
        model = AttentionRCNN(num_classes, args.dropout)
    else:
        print("Choose a Model!")
        model = None
        exit(1)
    config['model'] = {'name': args.model}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    config['optimizer'] = 'Adam'

    result_dir = "./" + args.dataset + "_results"
    if not os.path.exists(result_dir):
        os.makedirs("./" + args.dataset + "_results")
    with open(os.path.join(result_dir, f"config.yaml"), "w") as fp:
        yaml.dump(config, fp)


    # init rest
    max_f1 = -1.
    best_epoch = 0
    best_results = {}
    best_class_results = {}
    best_state = None
    already_trained = False

    writer = SummaryWriter(log_dir="./" + args.dataset + "_results/tensorboard")

    # training
    for epoch in range(1, args.epochs + 1):

        # check if model was already trained and load it in that case
        if os.path.exists(f'{args.dataset}_results/weights_{args.fold}.pth'):
            print("Model was already trained. Load best state and skip to evaluation")
            best_state = torch.load(f'{args.dataset}_results/weights_{args.fold}.pth')
            already_trained = True
            break

        model.to(device)
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
        dev_results, dev_class_results = evaluate(
            model,
            device,
            dev_loader,
            id2cls,
            decision_threshold=args.decision_threshold,
            apply_sigmoid=(args.model != 'Baseline')
        )
        print(f"dev results at epoch {epoch}:\n{yaml.dump(dev_results)}")
        writer.add_scalar('F1-Score/dev', dev_results['f1'], epoch)
        for cls in cls2id.keys():
            writer.add_scalar(f'F1-Score_({cls})', dev_class_results[cls]['f1'], epoch)

        # show and save results
        if dev_results['f1'] > max_f1:
            max_f1 = dev_results['f1']
            best_epoch = epoch
            best_state = model.cpu().state_dict().copy()
            best_results = dev_results.copy()
            best_class_results = dev_class_results.copy()

        if args.early_stopping is not None and epoch - best_epoch > args.early_stopping:
            print(f"No improvements for more than {args.early_stopping} epochs. Stopping here...")
            break

    if not already_trained:
        print(f"Best dev results found at epoch {best_epoch + 1}:\n{yaml.dump(best_results)}")
        best_results["Epoch"] = best_epoch + 1
        with open(os.path.join(result_dir, f"dev_fold{args.fold}.yaml"), "w") as f:
            yaml.dump(best_results, f)
            yaml.dump(best_class_results, f)

        torch.save(best_state, os.path.join(
                result_dir, f"weights_{args.fold}.pth"))

    model.load_state_dict(best_state)
    model.eval()
    test_results, test_class_results = evaluate(
        model,
        device,
        test_loader,
        id2cls,
        decision_threshold=args.decision_threshold,
        apply_sigmoid=(args.model != 'Baseline')
    )
    print(f"Best test results:\n{yaml.dump(test_results)}")
    with open(os.path.join(result_dir, f"test_fold{args.fold}.yaml"), "w") as fp:
        yaml.dump(test_results, fp)
        yaml.dump(test_class_results, fp)

    print("\nCalculate predictions on test set...")
    # get predictions on test set
    # get every test file
    test_files = get_test_files(args.dataset_location, args.dataset, fold=args.fold)

    predictions = []
    for test_file in tqdm.tqdm(test_files):
        # turn audio to features ready for model
        features = extract_mel_spectrograms([test_file], args.dataset_location, args.dataset, args.clip_length,
                                            args.n_fft, args.n_mels, hop_length, win_length)

        features = list(features.values())

        # pass features to model
        raw_prediction = []
        for feature in features:
            feature = torch.from_numpy(feature).float().unsqueeze(0).to(device)
            output = model(feature)
            raw_prediction.append(output.squeeze())

        raw_prediction = torch.cat(raw_prediction, dim=0)

        test_file_path = args.dataset_location + args.dataset + '/' + test_file
        audio_duration = librosa.get_duration(path=test_file_path)

        # turn output into appropriate labels with this format:
        prediction = get_prediction_from_raw_output(raw_prediction, id2cls, audio_duration, args.block_length,
                                                    test_file, decision_threshold=args.decision_threshold,
                                                    apply_sigmoid = (args.model != 'Baseline'))

        predictions += prediction

    print("Saving predictions...")
    # save the predicted labels
    pred_df = pd.DataFrame(predictions, columns=['filename', 'onset', 'offset', 'event_label'])
    pred_df.to_csv(f'{args.dataset}_results/test_predictions.tsv', sep='\t', index=False)