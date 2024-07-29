import glob
import os
import wave

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

class_weights = {
    'TUT': [0.8, 0.5, 0.85, 0.95, 0.85, 0.9],
    'BirdSED': torch.ones(92, dtype=torch.float)
}

dataset_ratio = {
    'TUT': torch.tensor(5.),
    'desed_2022': torch.tensor(50.),
    'BirdSED': torch.tensor(100.)
}

metadata2filepath = {
    'metadata/train/audioset_strong.tsv': 'audio/train/strong_label_real_16k/',
    'metadata/train/synthetic21_train/soundscapes.tsv': 'audio/train/synthetic21_train/soundscapes_16k/',
    'metadata/Ground-truth/mapped_ground_truth_eval.tsv': 'audio/eval21_16k/',
    'metadata/validation/eval_dcase2018.tsv': 'audio/validation/',
    'metadata/validation/validation.tsv': 'audio/validation/validation_16k/',
    'metadata/validation/synthetic21_validation/soundscapes.tsv': 'audio/validation/synthetic21_validation/soundscapes_16k/',
    'metadata/train/weak.tsv': 'audio/train/weak_16k/'
}


def get_classes(dataset, dataset_location, fold):
    if dataset == 'TUT':
        meta_data_path = dataset_location + dataset + '/meta.txt'
        data = pd.read_csv(meta_data_path, sep='\t',
                              names=['file', 'location', 'onset', 'offset', 'event', 'type', 'audio'], header=None)
        classes = data['event'].unique().tolist()
    elif dataset == 'desed_2022':
        meta_data_path = dataset_location + dataset + '/metadata/train/audioset_strong.tsv'
        data = pd.read_csv(meta_data_path, sep='\t')
        classes = data['event_label'].unique().tolist()
    elif dataset == 'BirdSED':
        # the fold determines which variant of BirdSED is done
        # 1 = Normal
        # 2 = Top Ten classes
        # 3 = Single label per file
        # 4 = Binary BirdSED
        if fold in [1, 3]:
            classes = []
            with open(dataset_location + dataset + '/bird_classes.txt') as f:
                    for line in f.readlines():
                        classes.append(line.replace('\n', ''))
        elif fold == 2:
            classes = ['Carduelis_chloris', 'Milvus_migrans', 'Lanius_collurio', 'Columba_palumbus',
                       'Coturnix_coturnix', 'Phylloscopus_sibilatrix', 'Passer_domesticus', 'Dendrocopos_major',
                       'Loxia_curvirostra', 'Lullula_arborea']
        else:
            classes = ['bird']
    else:
        classes = None

    return classes


def binarize_labels(labels, num_classes):
    bl = np.zeros(num_classes)
    bl[labels] = 1

    return bl


def get_splits(dataset, dataset_location, fold=None, use_weak=False, use_unlabelled=False):
    dataset_dir = dataset_location + dataset

    if dataset == 'TUT':
        split_pattern = dataset_dir + f'/evaluation_setup/street_fold{fold}_'
        splits = []
        meta_file = []
        for split in ['train', 'evaluate', 'test']:
            content = pd.read_csv(split_pattern + split + '.txt', sep='\t',
                                  names=['file', 'location', 'onset', 'offset', 'event'], header=None)
            splits.append(content)
            meta_file.append(split_pattern + split + '.txt')
        train_set, dev_set, test_set = splits
        train_files, dev_files, test_files = (train_set['file'].unique().tolist(), dev_set['file'].unique().tolist(),
                                              test_set['file'].unique().tolist())
        train_files = [filename for filename in train_files]
        dev_files = [filename for filename in dev_files]
        test_files = [filename for filename in test_files]
        train_labels, dev_labels, test_labels = meta_file
        test_labels = dev_labels
    elif dataset == 'desed_2022':
        train_files, dev_files, test_files = [], [], []
        train_labels, dev_labels, test_labels = [], [], []

        # for train
        # get strong real
        train_files += glob.glob(dataset_dir + '/audio/train/strong_label_real_16k/*.wav')
        train_labels.append(dataset_dir + '/metadata/train/audioset_strong.tsv')

        # get strong synthetic
        train_files += glob.glob(dataset_dir + '/audio/train/synthetic21_train/soundscapes_16k/*.wav')
        train_labels.append(dataset_dir + '/metadata/train/synthetic21_train/soundscapes.tsv')
        train_files = [file.replace(dataset_dir + '/', '', 1) for file in train_files]

        # get weak
        if use_weak:
            pass

        # get unlabelled
        if use_unlabelled:
            pass
        # these currently go unused as the model can only be trained on strongly labelled data

        # for dev
        # get strong real
        # dev_files += glob.glob(dataset_dir + '/audio/validation/*.wav')
        # dev_labels.append(dataset_dir + '/metadata/validation/eval_dcase2018.tsv')

        dev_files += glob.glob(dataset_dir + '/audio/validation/validation_16k/*.wav')
        dev_labels.append(dataset_dir + '/metadata/validation/validation.tsv')

        # get strong synthetic
        dev_files += glob.glob(dataset_dir + '/audio/validation/synthetic21_validation/soundscapes_16k/*.wav')
        dev_labels.append(dataset_dir + '/metadata/validation/synthetic21_validation/soundscapes.tsv')
        dev_files = [file.replace(dataset_dir + '/', '', 1) for file in dev_files]

        # for test
        # get strong real
        test_files += glob.glob(dataset_dir + '/audio/eval21_16k/*.wav')
        test_labels.append(dataset_dir + '/metadata/Ground-truth/mapped_ground_truth_eval.tsv')
        test_files = [file.replace(dataset_dir + '/', '', 1) for file in test_files]
    elif dataset == 'BirdSED':
        if fold == 1:
            soundscapes = 'soundscapes'
            metadata = 'metadata'
            split = '/splits/stratified'
        elif fold == 2:
            soundscapes = 'soundscapes'
            metadata = 'metadata'
            split = '/splits/top_ten'
        elif fold == 3:
            soundscapes = 'single_label_soundscapes'
            metadata = 'metadata'
            split = '/splits/single_label'
        else:
            soundscapes = 'soundscapes'
            metadata = 'binary_metadata'
            split = '/splits'

        with open(dataset_dir + split + '/train.txt', 'r') as f:
            train_files = [soundscapes + '/audio/' + file.replace('\n', '') for file in f.readlines()]
            train_labels = [dataset_location + dataset + '/' +  file.replace('audio', metadata).replace('.wav', '.txt') for file in train_files]

        with open(dataset_dir + split + '/val.txt', 'r') as f:
            dev_files = [soundscapes + '/audio/' + file.replace('\n', '') for file in f.readlines()]
            dev_labels = [dataset_location + dataset + '/' + file.replace('audio', metadata).replace('.wav', '.txt') for file in dev_files]

        with open(dataset_dir + split + '/test.txt', 'r') as f:
            test_files = [soundscapes + '/audio/' + file.replace('\n', '') for file in f.readlines()]
            test_labels = [dataset_location + dataset + '/' + file.replace('audio', metadata).replace('.wav', '.txt') for file in test_files]
    else:
        train_files, dev_files, test_files = [], [], []
        train_labels, dev_labels, test_labels = [], [], []

    return (train_files, train_labels), (dev_files, dev_labels), (test_files, test_labels)


def get_binary_labels(metadata_files, dataset_location, dataset, fold, clip_length, block_length, cls2id, save_location):
    clip_length = 0.001 * clip_length  # ms to s
    block_length = 0.001 * block_length  # ms to s

    labels = {}
    for metadata_file in tqdm(metadata_files, disable=(len(metadata_files) == 1)):

        if dataset == 'TUT':
            metadata = pd.read_csv(metadata_file, sep='\t',
                               names=['filename', 'location', 'onset', 'offset', 'event'], header=None)
            filenames = metadata['filename'].unique().tolist()
            event_keys = ['onset', 'offset', 'event']
        elif dataset == 'desed_2022':
            metadata = pd.read_csv(metadata_file, sep='\t')
            filenames = metadata['filename'].unique().tolist()
            event_keys = ['onset', 'offset', 'event_label']
        elif dataset == 'BirdSED':
            metadata = pd.read_csv(metadata_file, sep='\t', names=['onset', 'offset', 'event'], header=None)
            single_label = 'single_label_' if fold == 3 else ''
            filenames = [single_label + 'soundscapes/audio/' + metadata_file.split('/')[-1]]
            event_keys = ['onset', 'offset', 'event']
        else:
            metadata = None
            filenames = None
            event_keys = None

        for filename in tqdm(filenames, disable=(len(filenames) == 1)):
            if dataset == 'TUT':
                audio_path = dataset_location + dataset + '/' + filename
                audio_length = librosa.get_duration(path=audio_path)
                file_metadata = metadata.loc[metadata['filename'] == filename]
                file_path = filename
            elif dataset == 'desed_2022':
                audio_length = 10.
                file_metadata = metadata.loc[metadata['filename'] == filename]
                file_path = metadata_file[metadata_file.find('metadata'):]
                file_path = metadata2filepath[file_path] + filename
            elif dataset == 'BirdSED':
                audio_length = 10.
                file_metadata = metadata
                file_path = filename.replace('.txt', '.wav')
            else:
                audio_length = 0.
                file_metadata = None
                file_path = None

            for i in range(int(np.ceil(audio_length / clip_length))):
                onset_clip = i * clip_length

                clip_events = get_clip_events(file_metadata, clip_length, block_length, event_keys,
                                              cls2id, onset_clip=onset_clip)

                label = np.stack(clip_events)

                file_path_no_slashes = file_path.replace('/', '_slash_')
                label_file_path = save_location + '/' + file_path_no_slashes + '_' + str(i) + '.npy'
                np.save(label_file_path, label)

                labels[(i, file_path)] = label_file_path


    return labels


def get_weak_labels(metadata_files, cls2id):
    num_classes = len(cls2id.keys())

    labels = {}
    for metadata_file in metadata_files:
        metadata = pd.read_csv(metadata_file, sep='\t')

        for i, row in tqdm(metadata.iterrows()):
            events = row['event_labels'].split(',')
            events = [cls2id[cls] for cls in events]
            file_name = row['filename']
            file_path = metadata_file[metadata_file.find('metadata'):]
            file_path = metadata2filepath[file_path] + file_name

            binary_events = binarize_labels(events, num_classes)

            labels[(0, file_path)] = binary_events

    return labels


def get_clip_events(metadata, clip_length, block_length, columns, cls2id, onset_clip=0):
    onset, offset, event = columns
    num_classes = len(list(cls2id.keys()))
    metadata = metadata.sort_values(by=[onset])

    # clip_length = 0.001 * clip_length  # ms to s
    # block_length = 0.001 * block_length  # ms to s

    clip_events = []
    for j in range(int(clip_length // block_length) + 1):
        onset_block = onset_clip + j * block_length
        offset_block = onset_clip + (j + 1) * block_length

        block_events = []
        for index, row in metadata.iterrows():
            if onset_block <= row[onset] < offset_block or onset_block < row[offset] <= offset_block:
                block_events.append(cls2id[row[event]])
            elif onset_block >= row[onset] and offset_block <= row[offset]:
                block_events.append(cls2id[row[event]])

            if row[onset] > offset_block:
                break

        binary_labels = binarize_labels(list(set(block_events)), num_classes)
        clip_events.append(binary_labels)

    return clip_events


def get_test_files(dataset_location, dataset, fold=None):
    if dataset == 'TUT':
        assert fold is not None
        file_path = dataset_location + f'TUT/evaluation_setup/street_fold{fold}_test.txt'
        test_df = pd.read_csv(file_path, sep='\t', names=['filename', 'scene'], header=None)
        test_files = test_df['filename'].tolist()
    elif dataset == 'desed_2022':
        file_path = dataset_location + f'desed_2022/audio/eval21_16k/*.wav'
        test_files = glob.glob(file_path)
        test_files = [file.replace(dataset_location + 'desed_2022/', '') for file in test_files]
    elif dataset == 'BirdSED':
        if fold == 1:
            with open(dataset_location + dataset + '/splits/stratified/test.txt', 'r') as f:
                test_files = ['soundscapes/audio/' + file.replace('\n', '') for file in f.readlines()]
        elif fold == 2:
            with open(dataset_location + dataset + '/splits/top_ten/test.txt', 'r') as f:
                test_files = ['soundscapes/audio/' + file.replace('\n', '') for file in f.readlines()]
        elif fold == 3:
            with open(dataset_location + dataset + '/splits/single_label/test.txt', 'r') as f:
                test_files = ['single_label_soundscapes/audio/' + file.replace('\n', '') for file in f.readlines()]
        else:
            with open(dataset_location + dataset + '/splits/test.txt', 'r') as f:
                test_files = ['soundscapes/audio/' + file.replace('\n', '') for file in f.readlines()]
    else:
        test_files = None

    return test_files


def get_labels(name, dataset_location, dataset, fold, clip_length, block_length, cls2id, metadata_location=None, metadata_files=None, weak=False):
    dir_path = dataset_location + dataset + '/' + name + f'_{fold}_{clip_length}_{block_length}'

    if os.path.isdir(dir_path):
        labels = {}
        print('getting labels dict from ' + dir_path + "...")
        for filename in glob.glob(dir_path + '/*.npy'):
            key_name = filename.replace('\\', '/').split('/')[-1]
            labels[key_name] = filename
    else:
        os.makedirs(dir_path, exist_ok=True)

        if metadata_location is not None:
            label_file = dataset_location + '/' + dataset + '/' + metadata_location
            print('extracting labels from ' + label_file + "...")
            label_files = [label_file]
        elif metadata_files is not None:
            print('extracting and saving labels from given metadata_files...')
            if type(metadata_files) is str:
                label_files = [metadata_files]
            else:
                label_files = metadata_files
        else:
            label_files = None
            # TODO

        if not weak:
            labels = get_binary_labels(label_files, dataset_location, dataset, fold, clip_length, block_length, cls2id, dir_path)
        else:
            labels = get_weak_labels(label_files, dataset, cls2id)

        print("Done!")

    return labels


def get_single_label(labels):
    new_labels = {}
    for key, value in labels.items():
        cls = np.max(value, axis=0)

        binary_labels = np.max(value, axis=1)

        new_labels[key] = (binary_labels, cls)

    return new_labels
