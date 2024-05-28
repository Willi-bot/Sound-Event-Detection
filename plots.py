import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import get_wav_duration

def plot_class_percentages(dataset_dir):
    audio_file_paths = glob.glob(os.path.join(dataset_dir, "audio/street/*.wav"))
    metadata_file_paths = glob.glob(os.path.join(dataset_dir, "meta/street/*.ann"))

    total_length = 0
    class_length = {}
    for audio_file_path, metadata_file_path in zip(audio_file_paths, metadata_file_paths):
        audio_file_path = audio_file_path.replace("\\", "/")
        file_name = "/".join(audio_file_path.split('/')[-3:])

        metadata = pd.read_csv(metadata_file_path, sep='\t',
                              names=['onset', 'offset', 'event'], header=None)

        audio_length = get_wav_duration(audio_file_path)
        total_length += audio_length

        for i, row in metadata.iterrows():
            event_length = row['offset'] - row['onset']
            if row['event'] in class_length:
                class_length[row['event']] += event_length
            else:
                class_length[row['event']] = event_length

    classes = list(class_length.keys())
    event_lengths = list(class_length.values())
    plt.bar(classes, event_lengths, color='mediumspringgreen')
    plt.xlabel('Classes')
    plt.ylabel('Total length in seconds')
    plt.title('Total length distribution')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.show()

    percentages = [event_length / total_length for event_length in event_lengths]
    plt.bar(classes, percentages, color='mediumspringgreen')
    plt.xlabel('Classes')
    plt.ylabel('Percentage')
    plt.title('Percentage Distribution')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.show()


def print_pos_neg_ratio(dataset, label_file):
    labels = np.load(label_file, allow_pickle=True).item()

    flattened_labels = []
    for value in labels.values():
        flattened_labels.append(value)

    labels = np.stack(flattened_labels).flatten()

    positives = np.sum(labels)
    negatives = len(labels) - positives

    print(f"Dataset: {dataset}")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")
    print(f"Ratio Negative to Positives: {negatives / positives}\n")


if __name__ == '__main__':
    plot_class_percentages("dataset/TUT")

    print_pos_neg_ratio("TUT", "dataset/TUT/labels/labels_10000_40.npy")

    print_pos_neg_ratio("desed_2022", "dataset/desed_2022/labels/labels_10000_40.npy")