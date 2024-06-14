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
from utils import get_classes, get_splits, get_test_files, get_labels
from data_loading import get_dataloader
from feature_extraction import sampling_rates, get_features, extract_mel_spectrograms
from models import AdvancedRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SED Semisupervised")
    parser.add_argument(
        "--dataset",
        help="name of the dataset to use",
        default='desed_2022',
        choices=['desed_2022']
    )
    parser.add_argument(
        '--iterations',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--val-step',
        default=1000,
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
        default=16,
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
        '--normalize',
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
        '--ema-decay',
        default=0.999,
        type=float
    )
    parser.add_argument(
        '--use-hard-labels',
        action='store_true'
    )
    parser.add_argument(
        '--unlabelled-threshold',
        default=0.4,
        type=float
    )
    args = parser.parse_args()

    config = vars(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_dir = args.dataset_location + args.dataset

    # assignment of files to each split
    train_files, dev_files, test_files = get_splits(args.dataset, args.dataset_location)

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

    # get strong features and labels
    print("\nGet features and labels of real strongly annotated data...")
    strong_train_features = get_features("features/strong_train", args.dataset_location, args.dataset,
                                         args.clip_length, args.n_fft, args.n_mels, hop_length, win_length,
                                         'audio/train/strong_label_real_16k/*.wav')

    strong_train_labels = get_labels('labels/strong_train', args.dataset_location, args.dataset, args.clip_length,
                        args.block_length, cls2id, 'metadata/train/audioset_strong.tsv')

    # get strong synthetic features and labels
    print("\nGet features and labels of synthetic strongly annotated data...")
    strong_train_features = strong_train_features | get_features("features/synthetic_train", args.dataset_location, args.dataset,
                                         args.clip_length, args.n_fft, args.n_mels, hop_length, win_length,
                                         'audio/train/synthetic21_train/soundscapes/*.wav')

    strong_train_labels = strong_train_labels | get_labels('labels/synthetic_train', args.dataset_location, args.dataset, args.clip_length,
                        args.block_length, cls2id, 'metadata/train/synthetic21_train/soundscapes.tsv')

    strong_train_dataloader = get_dataloader(strong_train_features, strong_train_labels, args.batch_size, shuffle=True, drop_last=True,
                                             use_specaug=args.use_specaug)

    # get weakly annotated
    print("\nGet features and labels of real weakly annotated data...")
    weak_train_features = get_features("features/weak_train", args.dataset_location, args.dataset,
                                         args.clip_length, args.n_fft, args.n_mels, hop_length, win_length,
                                       "audio/train/weak_16k/*.wav")
    weak_train_labels = get_labels('labels/weak_train', args.dataset_location, args.dataset, args.clip_length,
                        args.block_length, cls2id, 'metadata/train/weak.tsv', weak=True)

    weak_train_dataloader = get_dataloader(weak_train_features, weak_train_labels, args.batch_size, shuffle=True, drop_last=True,
                                           use_specaug=args.use_specaug)

    # get unannotated data
    print("\nGet features of unannotated data...")
    unlabelled_train_features = get_features("features/unlabbelled_train", args.dataset_location, args.dataset,
                                             args.clip_length, args.n_fft, args.n_mels, hop_length, win_length,
                                             "audio/train/unlabel_in_domain_16k/*.wav")

    unlabelled_train_dataloader = get_dataloader(unlabelled_train_features, unlabelled_train_features, args.batch_size, shuffle=True, drop_last=True,
                                           use_specaug=args.use_specaug)

    # get validation data
    print("\nGet features and labels of validation data...")
    strong_eval_features = get_features("features/strong_eval", args.dataset_location, args.dataset,
                                             args.clip_length, args.n_fft, args.n_mels, hop_length, win_length,
                                        "audio/validation/validation_16k/*.wav")
    strong_eval_labels = get_labels("labels/strong_eval", args.dataset_location, args.dataset, args.clip_length,
                                    args.block_length, cls2id, "metadata/validation/validation.tsv")

    strong_eval_features = strong_eval_features | get_features("features/strong_eval_2018", args.dataset_location, args.dataset,
                                             args.clip_length, args.n_fft, args.n_mels, hop_length, win_length,
                                         "audio/validation/*.wav")
    strong_eval_labels = strong_eval_labels | get_labels("labels/strong_eval_2018", args.dataset_location, args.dataset,
                                     args.clip_length, args.block_length, cls2id,
                                     "metadata/validation/eval_dcase2018.tsv")

    strong_eval_features = strong_eval_features | get_features("features/synthetic_eval", args.dataset_location, args.dataset,
                                         args.clip_length, args.n_fft, args.n_mels, hop_length, win_length,
                                         "audio/validation/synthetic21_validation/soundscapes_16k/*.wav")
    strong_eval_labels = strong_eval_labels | get_labels("labels/synthetic_eval", args.dataset_location, args.dataset,
                                     args.clip_length, args.block_length, cls2id,
                                     "metadata/validation/synthetic21_validation/soundscapes.tsv")

    strong_eval_dataloader = get_dataloader(strong_eval_features, strong_eval_labels, args.batch_size, shuffle=False,
                                            drop_last=False, use_specaug=False)

    # initialize model & training stuff (optimizer, scheduler, loss func...)
    strong_loss_fn = torch.nn.BCEWithLogitsLoss()
    weak_loss_fn = torch.nn.BCEWithLogitsLoss()


    student_model = AdvancedRCNN(num_classes, args.dropout).to(device)
    config['model'] = {'name': "AdvancedRCNN"}
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    config['optimizer'] = 'Adam'

    teacher_model = AdvancedRCNN(num_classes).to(device)

    result_dir = "./" + args.dataset + "_semisupervised_results"
    if not os.path.exists(result_dir):
        os.makedirs("./" + args.dataset + "_semisupervised_results")
    with open(os.path.join(result_dir, f"config.yaml"), "w") as fp:
        yaml.dump(config, fp)


    # init rest
    max_f1 = 0
    best_step = 0
    best_results = {}
    best_class_results = {}
    delta = 0.
    ema_rate = args.ema_decay
    best_student_state = None
    best_teacher_state = None
    already_trained = False
    unlabelled_f1 = 0.

    strong_train_iter = iter(strong_train_dataloader)
    weak_train_iter = iter(weak_train_dataloader)
    unlabelled_train_iter = iter(unlabelled_train_dataloader)

    writer = SummaryWriter(log_dir="./" + args.dataset + "_semisupervised_results/tensorboard")

    # training in iteration steps because dataloader are not equally big
    for step in tqdm.tqdm(range(1, args.iterations + 1)):

        # check if model was already trained and load it in that case
        if os.path.exists(f'{args.dataset}_semisupervised_results/weights.pth'):
            print("Model was already trained. Load best state and skip to evaluation")
            best_student_state = torch.load(f'{args.dataset}_semisupervised_results/student_weights.pth')
            best_teacher_state = torch.load(f'{args.dataset}_semisupervised_results/teacher_weights.pth')
            already_trained = True
            break

        # get data from batches
        try:
            strong_x, strong_y = next(strong_train_iter)
        except StopIteration:
            strong_train_iter = iter(strong_train_dataloader)
            strong_x, strong_y = next(strong_train_iter)

        try:
            weak_x, weak_y = next(weak_train_iter)
        except StopIteration:
            weak_train_iter = iter(weak_train_dataloader)
            weak_x, weak_y = next(weak_train_iter)

        try:
            unlabelled_x, useless_labels = next(unlabelled_train_iter)
        except StopIteration:
            unlabelled_train_iter = iter(unlabelled_train_dataloader)
            unlabelled_x, useless_labels = next(unlabelled_train_iter)

        # get pseudo labels
        teacher_model.train()
        teacher_model = teacher_model.to(device)
        teacher_output = teacher_model(unlabelled_x.to(device))
        pseudo_labels = torch.sigmoid(teacher_output)

        if args.use_hard_labels:
            pseudo_labels = torch.where(teacher_output > args.decision_threshold, 1., 0.).to(device)

        student_model.to(device)
        student_model.train()

        strong_y_hat = student_model(strong_x.to(device))
        strong_y_hat, strong_y = strong_y_hat.reshape(-1, num_classes), strong_y.reshape(-1, num_classes).to(device)
        loss = strong_loss_fn(strong_y_hat, strong_y)

        weak_y_hat = student_model(weak_x.to(device))
        weak_y_hat = student_model.get_weak_logits(weak_y_hat)
        loss += weak_loss_fn(weak_y_hat, weak_y.to(device))

        pseudo_y_hat = student_model(unlabelled_x.to(device))
        pseudo_y_hat, pseudo_labels = pseudo_y_hat.reshape(-1, num_classes), pseudo_labels.reshape(-1, num_classes)
        loss += delta * strong_loss_fn(pseudo_y_hat, pseudo_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher
        with torch.no_grad():
            for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                teacher_param.data = ema_rate * teacher_param.data + (1 - ema_rate) * student_param.data

            for student_buffer, teacher_buffer in zip(student_model.buffers(), teacher_model.buffers()):
                teacher_buffer.data = ema_rate * teacher_buffer.data + (1 - ema_rate) * student_buffer.data

        teacher_model.eval()
        if unlabelled_f1 < args.unlabelled_threshold and (step % int(args.val_step // 10)) == 0:
            teacher_results, _ = evaluate(
                teacher_model,
                device,
                strong_eval_dataloader,
                id2cls
            )
            unlabelled_f1 = teacher_results['f1']
            if unlabelled_f1 > args.unlabelled_threshold:
                delta = 1.
            else:
                delta = 0.

        if (step % args.val_step) == 0:
            student_model.eval()
            dev_results, dev_class_results = evaluate(
                student_model,
                device,
                strong_eval_dataloader,
                id2cls
            )
            print(f"dev results at step {step}:\n{yaml.dump(dev_results)}")
            writer.add_scalar('F1-Score/dev', dev_results['f1'], step)
            for cls in cls2id.keys():
                writer.add_scalar(f'F1-Score_({cls})', dev_class_results[cls]['f1'], step)

            # show and save results
            if dev_results['f1'] > max_f1:
                max_f1 = dev_results['f1']
                best_step = step
                best_student_state = student_model.cpu().state_dict().copy()
                best_teacher_state = teacher_model.cpu().state_dict().copy()
                best_results = dev_results.copy()
                best_class_results = dev_class_results.copy()

    if not already_trained:
        print(f"Best dev results found at epoch {best_step + 1}:\n{yaml.dump(best_results)}")
        best_results["Step"] = best_step + 1
        with open(os.path.join(result_dir, f"dev.yaml"), "w") as f:
            yaml.dump(best_results, f)
            yaml.dump(best_class_results, f)

        torch.save(best_student_state, os.path.join(
                result_dir, "student_weights.pth"))
        torch.save(best_teacher_state, os.path.join(
                result_dir, "student_teacher_weights.pth"))

    student_model.load_state_dict(best_student_state)
    student_model.eval()

    print("\nCalculate predictions on test set for student model...")
    # get predictions on test set
    # get every test file
    test_files = get_test_files(args.dataset)

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
            output = student_model(feature)
            raw_prediction.append(output.squeeze())

        raw_prediction = torch.cat(raw_prediction, dim=0)

        test_file_path = args.dataset_location + args.dataset + '/' + test_file
        audio_duration = librosa.get_duration(path=test_file_path)

        # turn output into appropriate labels with this format:
        prediction = get_prediction_from_raw_output(raw_prediction, id2cls, audio_duration, args.block_length,
                                                    test_file, decision_threshold=args.decision_threshold)

        predictions += prediction

    print("Saving predictions...")
    # save the predicted labels
    pred_df = pd.DataFrame(predictions, columns=['filename', 'onset', 'offset', 'event_label'])
    pred_df.to_csv(f'{args.dataset}_semisupervised_results/student_test_predictions.tsv', sep='\t', index=False)

    teacher_model.load_state_dict(best_teacher_state)
    teacher_model.eval()

    print("\nCalculate predictions on test set for teacher model...")
    # get predictions on test set
    # get every test file
    test_files = get_test_files(args.dataset)

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
            output = teacher_model(feature)
            raw_prediction.append(output.squeeze())

        raw_prediction = torch.cat(raw_prediction, dim=0)

        test_file_path = args.dataset_location + args.dataset + '/' + test_file
        audio_duration = librosa.get_duration(path=test_file_path)

        # turn output into appropriate labels with this format:
        prediction = get_prediction_from_raw_output(raw_prediction, id2cls, audio_duration, args.block_length,
                                                    test_file, decision_threshold=args.decision_threshold)

        predictions += prediction

    print("Saving predictions...")
    # save the predicted labels
    pred_df = pd.DataFrame(predictions, columns=['filename', 'onset', 'offset', 'event_label'])
    pred_df.to_csv(f'{args.dataset}_semisupervised_results/teacher_test_predictions.tsv', sep='\t', index=False)