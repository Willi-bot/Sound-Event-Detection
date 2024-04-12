import glob
import os
import wave
import numpy as np
import pandas as pd


def get_classes(dataset):
    meta_data_path = 'dataset/' + dataset + '/meta.txt'
    data = pd.read_csv(meta_data_path, sep='\t',
                          names=['file', 'location', 'onset', 'offset', 'event', 'type', 'audio'], header=None)

    return data['event'].unique().tolist()


def get_binary_event_labels(dataset_dir, clip_length, block_length, cls2id):
    audio_file_paths = glob.glob(os.path.join(dataset_dir, "audio/street/*.wav"))
    metadata_file_paths = glob.glob(os.path.join(dataset_dir, "meta/street/*.ann"))
    num_classes = len(list(cls2id.keys()))

    clip_length = 0.001 * clip_length # ms to s
    block_length = 0.001 * block_length # ms to s

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

            clip_events = []
            for j in range(int(clip_length // block_length) + 1):
                onset_block = onset_clip + j * block_length
                offset_block = onset_clip + (j + 1) * block_length

                block_events = []
                for index, row in metadata.iterrows():
                    if row['offset'] < onset_block:
                        continue

                    if onset_block < row['onset'] < offset_block or onset_block < row['offset'] < offset_block:
                        block_events.append(cls2id[row['event']])
                    elif onset_block > row['onset'] and offset_block < row['offset']:
                        block_events.append(cls2id[row['event']])

                    if row['onset'] > offset_block:
                        break

                binary_labels = get_binary_labels(list(set(block_events)), num_classes)
                clip_events.append(binary_labels)

            labels[(i, file_name)] = np.stack(clip_events)
            all_events += clip_events.copy()

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


def get_splits(dataset, fold):
    dataset_dir = 'dataset/' + dataset
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

    return train_files, dev_files, test_files


def get_split_data(split, features, labels):

    split_features = []
    split_labels = []

    for data_point in split:
        for key in features:
            if data_point == key[1]:
                split_features.append(features[key])
                split_labels.append(labels[key])

    return split_features, split_labels



