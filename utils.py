import glob
import os
import wave
import numpy as np
import pandas as pd
import torch


class_weights = {
    'TUT': [0.8, 0.5, 0.85, 0.95, 0.85, 0.9]
}

dataset_ratio = {
    'TUT': torch.tensor(5.),
    'desed_2022': torch.tensor(50.)
}


def get_classes(dataset, dataset_location):
    if dataset == 'TUT':
        meta_data_path = dataset_location + dataset + '/meta.txt'
        data = pd.read_csv(meta_data_path, sep='\t',
                              names=['file', 'location', 'onset', 'offset', 'event', 'type', 'audio'], header=None)
        classes = data['event'].unique().tolist()
    elif dataset == 'desed_2022':
        meta_data_path = dataset_location + dataset + '/metadata/train/audioset_strong.tsv'
        data = pd.read_csv(meta_data_path, sep='\t')
        classes = data['event_label'].unique().tolist()
    else:
        classes = None

    return classes


def get_binary_event_labels(dataset, clip_length, block_length, cls2id, dataset_location):
    if dataset == 'TUT':
        dataset_dir = dataset_location + dataset
        labels = get_tut_labels(dataset_dir, clip_length, block_length, cls2id)
    elif dataset == 'desed_2022':
        metadata_path = dataset_location + dataset + '/metadata/'
        labels = get_desed_labels(metadata_path, clip_length, block_length, cls2id)
    else:
        labels = None

    return labels


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = num_frames / float(frame_rate)
        return duration


def get_binary_labels(labels, num_classes):
    bl = np.zeros(num_classes)
    bl[labels] = 1

    return bl


def get_splits(dataset, fold, dataset_location):
    if dataset == 'TUT':
        dataset_dir = dataset_location + dataset
        split_pattern = dataset_dir + f'/evaluation_setup/street_fold{fold}_'
        splits = []
        for split in ['train', 'evaluate', 'test']:
            content = pd.read_csv(split_pattern + split + '.txt', sep='\t',
                                  names=['file', 'location', 'onset', 'offset', 'event'], header=None)
            splits.append(content)
        train_set, dev_set, test_set = splits
        train_files = train_set['file'].unique().tolist()
        dev_files = dev_set['file'].unique().tolist()
        test_files = test_set['file'].unique().tolist()
    elif dataset == 'desed_2022':
        dataset_dir = dataset_location + dataset
        # train
        train_files = get_file_names(dataset_dir + '/audio/train/strong_label_real_16k')
        train_files += get_file_names(dataset_dir + '/audio/train/synthetic21_train/soundscapes_16k')
        # validation
        dev_files = get_file_names(dataset_dir + '/audio/validation/validation_16k')
        dev_files += get_file_names(dataset_dir + '/audio/validation/synthetic21_validation/soundscapes_16k')
        # test
        test_files = get_file_names(dataset_dir + '/audio/eval21')

        train_files, dev_files, test_files = 'train', 'validation', 'validation'
    else:
        train_files, dev_files, test_files = [], [], []

    return train_files, dev_files, test_files


def get_split_data(split, features=None, labels=None, dataset=None):

    if dataset == 'TUT':
        split_features = []
        split_labels = []

        for data_point in split:
            for key in features:
                if data_point == key[1]:
                    split_features.append(features[key])
                    split_labels.append(labels[key])
    elif dataset == 'desed_2022':
        split_features = []
        split_labels = []

        for key in features.keys():
            if split in key:
                split_features.append(features[key])
                split_labels.append(labels[key])
    else:
        split_features = None
        split_labels = None

    return split_features, split_labels


def normalize_features(feature_dict):
    # Convert the feature dictionary to a numpy array
    features = np.array(list(feature_dict.values()))

    # Calculate mean and standard deviation across all samples
    mean = np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)

    # Normalize each feature
    normalized_features = {}
    for key, value in feature_dict.items():
        normalized_features[key] = (value - mean) / std_dev

    return normalized_features


def get_file_names(file_directory):
    filenames = glob.glob(file_directory + '/*.wav')

    return filenames


def get_tut_labels(dataset_dir, clip_length, block_length, cls2id):
    audio_file_paths = glob.glob(os.path.join(dataset_dir, "audio/street/*.wav"))
    metadata_file_paths = glob.glob(os.path.join(dataset_dir, "meta/street/*.ann"))


    clip_length = 0.001 * clip_length  # ms to s
    block_length = 0.001 * block_length  # ms to s

    all_events = []
    labels = {}
    for audio_file_path, metadata_file_path in zip(audio_file_paths, metadata_file_paths):
        audio_file_path = audio_file_path.replace("\\", "/")
        file_name = "/".join(audio_file_path.split('/')[-3:])

        metadata = pd.read_csv(metadata_file_path, sep='\t',
                               names=['onset', 'offset', 'event'], header=None)

        audio_length = get_wav_duration(audio_file_path)
        for i in range(int(audio_length // clip_length) + 1):
            onset_clip = i * clip_length

            clip_events = get_clip_events(metadata, clip_length, block_length, ['onset', 'offset', 'event'],
                                          cls2id, onset_clip=onset_clip)

            labels[(i, file_name)] = np.stack(clip_events)
            all_events += clip_events.copy()

    return labels


def get_desed_labels(metadata_path, clip_length, block_length, cls2id):
    tsv_paths = ['train/audioset_strong.tsv', 'train/synthetic21_train/soundscapes.tsv', 'validation/validation.tsv', 'validation/synthetic21_validation/soundscapes.tsv']

    labels = {}
    for path in tsv_paths:
        metadata = pd.read_csv(os.path.join(metadata_path, path), sep='\t')

        for file_name in metadata['filename'].unique().tolist():
            clip_events = get_clip_events(metadata.loc[metadata['filename'] == file_name], clip_length, block_length,
                                          ['onset', 'offset', 'event_label'], cls2id)

            key = ('train' if 'train' in path else 'validation') + ('/synthetic/' if 'synthetic' in path else '/strong/') + file_name
            labels[key] = np.array(clip_events)

    return labels


def get_clip_events(metadata, clip_length, block_length, columns, cls2id, onset_clip=0):
    onset, offset, event = columns
    num_classes = len(list(cls2id.keys()))

    clip_length = 0.001 * clip_length  # ms to s
    block_length = 0.001 * block_length  # ms to s

    clip_events = []
    for j in range(int(clip_length // block_length) + 1):
        onset_block = onset_clip + j * block_length
        offset_block = onset_clip + (j + 1) * block_length

        block_events = []
        for index, row in metadata.iterrows():
            if row[onset] < onset_block:
                continue

            if onset_block < row[onset] < offset_block or onset_block < row[offset] < offset_block:
                block_events.append(cls2id[row[event]])
            elif onset_block > row[onset] and offset_block < row[offset]:
                block_events.append(cls2id[row[event]])

            if row[onset] > offset_block:
                break

        binary_labels = get_binary_labels(list(set(block_events)), num_classes)
        clip_events.append(binary_labels)

    return clip_events
