import glob
import os

import librosa
import yaml
import argparse

import pandas as pd
from matplotlib import pyplot as plt

def plot_class_percentages(class_length, total_length, label):
    classes = list(class_length.keys())
    event_lengths = list(class_length.values())
    plt.bar(classes, event_lengths, color='mediumspringgreen')
    plt.xlabel('Classes')
    plt.ylabel('Total length in seconds')
    plt.title(f'Total length distribution ({label})')
    plt.xticks(rotation=90, fontsize='small')
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(f'plots/{label}_total.png')
    plt.show()

    percentages = [event_length / total_length for event_length in event_lengths]
    plt.bar(classes, percentages, color='mediumspringgreen')
    plt.xlabel('Classes')
    plt.ylabel('Percentage')
    plt.title(f'Percentage Distribution ({label})')
    plt.xticks(rotation=90, fontsize='small')
    plt.ylim(0, 1)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f'plots/{label}_percent.png')
    plt.show()


def plot_class_counts(class_counts, total_count, label):
    classes = list(class_counts.keys())
    event_counts = list(class_counts.values())
    plt.bar(classes, event_counts, color='mediumspringgreen')
    plt.xlabel('Classes')
    plt.ylabel('Total # of class appearances')
    plt.title(f'Total distribution ({label})')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(f'plots/{label}_total.png')
    plt.show()

    percentages = [event_count / total_count for event_count in event_counts]
    plt.bar(classes, percentages, color='mediumspringgreen')
    plt.xlabel('Classes')
    plt.ylabel('Percentage')
    plt.title(f'Percentage Distribution ({label})')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(f'plots/{label}_percent.png')
    plt.show()


def get_total_length(audio_files):
    total_length = 0
    for audio_file in audio_files:
        audio_length = librosa.get_duration(path=audio_file)
        total_length += audio_length

    return total_length


def get_class_lengths(audio_path, meta_path, dataset):

    total_length = get_total_length(audio_path)

    class_length = {}
    for meta_file in meta_path:
        if dataset == 'TUT' or dataset == 'Birds':
            metadata = pd.read_csv(meta_file, sep='\t',
                              names=['onset', 'offset', 'event'], header=None)
            onset_key = 'onset'
            offset_key = 'offset'
            event_key = 'event'
        elif dataset == 'DESED':
            metadata = pd.read_csv(meta_file, sep='\t')
            onset_key = 'onset'
            offset_key = 'offset'
            event_key = 'event_label'
        else:
            metadata = None
            onset_key = ''
            offset_key = ''
            event_key = ''

        for i, row in metadata.iterrows():
            event_length = row[offset_key] - row[onset_key]
            if row[event_key] in class_length:
                class_length[row[event_key]] += event_length
            else:
                class_length[row[event_key]] = event_length

    return class_length, total_length


def get_pos_neg_ratio(class_length, total_length):

    classwise_positives = class_length
    classwise_negatives = {cls: total_length - length for cls, length in class_length.items()}

    classwise_ratios = {cls: (float(classwise_negatives[cls]) / classwise_positives[cls]) for cls in classwise_positives.keys()}

    return classwise_ratios


def get_class_counts(meta_path):

    metadata = pd.read_csv(meta_path, sep='\t')

    class_counts = {}
    for i, row in metadata.iterrows():
        events = row['event_labels'].split(',')

        for event in events:
            if event in class_counts.keys():
                class_counts[event] += 1
            else:
                class_counts[event] = 1

    return class_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tut", action='store_true')
    parser.add_argument("--desed", action='store_true')
    parser.add_argument('--birds', action='store_true')

    args = parser.parse_args()

    if args.tut:
        # TUT part
        TUT_data_dict = {}

        dataset_dir = "dataset/TUT"
        audio_path = glob.glob(os.path.join(dataset_dir, "audio/street/*.wav"))
        TUT_data_dict["Audio File Count"] = len(audio_path)
        meta_path = glob.glob(os.path.join(dataset_dir, "meta/street/*.ann"))
        class_length, total_length = get_class_lengths(audio_path, meta_path, 'TUT')
        TUT_data_dict["Total length for each class (in s)"] = class_length
        TUT_data_dict["Total length of dataset (in s)"] = total_length
        plot_class_percentages(class_length, total_length, "TUT Data")

        ratios = get_pos_neg_ratio(class_length, total_length)
        TUT_data_dict["Ratio \'Class not present\' to \'Class present\' (for every class over every audio file)"] = ratios

        with open(os.path.join("plots/TUT/", f"TUT_stats.yaml"), "w") as fp:
            yaml.dump(TUT_data_dict, fp)

    if args.desed:
        # Desed part
        dataset_dir = "dataset/desed_2022"
        Desed_data_dict = {"Strongly Labeled": {}, "Weakly Labeled": {}, "Synthetic Data": {}, "Unlabeled Data": {}}

        # strongly labeled
        audio_path = glob.glob(os.path.join(dataset_dir, "audio/train/strong_label_real/*.wav"))
        Desed_data_dict["Strongly Labeled"]["Audio File Count"] = len(audio_path)
        meta_path = [os.path.join(dataset_dir, "metadata/train/audioset_strong.tsv")]
        class_length, total_length = get_class_lengths(audio_path, meta_path, 'DESED')
        Desed_data_dict["Strongly Labeled"]["Total length for each class (in s)"] = class_length
        Desed_data_dict["Strongly Labeled"]["Total length of dataset (in s)"] = total_length
        plot_class_percentages(class_length, total_length, "DESED Strongly Labeled")

        ratios = get_pos_neg_ratio(class_length, total_length)
        Desed_data_dict["Strongly Labeled"][
            "Ratio \'Class not present\' to \'Class present\' (for every class over every audio file)"] = ratios

        # Synthetic
        audio_path = glob.glob(os.path.join(dataset_dir, "audio/train/synthetic21_train/soundscapes/*.wav"))
        Desed_data_dict["Synthetic Data"]["Audio File Count"] = len(audio_path)
        meta_path = [os.path.join(dataset_dir, "metadata/train/synthetic21_train/soundscapes.tsv")]
        class_length, total_length = get_class_lengths(audio_path, meta_path, 'DESED')
        Desed_data_dict["Synthetic Data"]["Total length for each class (in s)"] = class_length
        Desed_data_dict["Synthetic Data"]["Total length of dataset (in s)"] = total_length
        plot_class_percentages(class_length, total_length, "DESED Synthetic Data")

        ratios = get_pos_neg_ratio(class_length, total_length)
        Desed_data_dict["Synthetic Data"][
            "Ratio \'Class not present\' to \'Class present\' (for every class over every audio file)"] = ratios

        # Weakly Labeled
        audio_path = glob.glob(os.path.join(dataset_dir, "audio/train/weak/*.wav"))
        meta_path = os.path.join(dataset_dir, "metadata/train/weak.tsv")
        total_length = get_total_length(audio_path)
        audio_file_count = len(audio_path)
        Desed_data_dict["Weakly Labeled"]["Audio File Count"] = audio_file_count
        class_counts = get_class_counts(meta_path)
        Desed_data_dict["Weakly Labeled"]["Total number of appearances per class"] = class_counts
        plot_class_counts(class_counts, audio_file_count, "DESED Weakly Labeled")

        ratios = get_pos_neg_ratio(class_counts, audio_file_count)
        Desed_data_dict["Weakly Labeled"][
            "Ratio \'Class not present\' to \'Class present\' (for every class over every audio file)"] = ratios

        # Unlabeled
        audio_path = glob.glob(os.path.join(dataset_dir, "audio/train/unlabel_in_domain/*.wav"))
        Desed_data_dict["Unlabeled Data"]["Audio File Count"] = len(audio_path)
        total_length = get_total_length(audio_path)
        Desed_data_dict["Unlabeled Data"]["Total Length of dataset (in s)"] = total_length

        with open(os.path.join("plots/Desed", f"Desed_stats.yaml"), "w") as fp:
            yaml.dump(Desed_data_dict, fp)

    if args.birds:
        bird_data_dict = {}

        dataset_dir = "dataset/bird_dataset"
        audio_path = glob.glob(os.path.join(dataset_dir, "soundscapes/audio/*.wav"))
        bird_data_dict["Audio File Count"] = len(audio_path)
        meta_path = glob.glob(os.path.join(dataset_dir, "soundscapes/metadata/*.txt"))
        class_length, total_length = get_class_lengths(audio_path, meta_path, 'Birds')
        bird_data_dict["Total length for each class (in s)"] = class_length
        bird_data_dict["Total length of dataset (in s)"] = total_length
        plot_class_percentages(class_length, total_length, "Synthetic Bird Data")

        ratios = get_pos_neg_ratio(class_length, total_length)
        bird_data_dict[
            "Ratio \'Class not present\' to \'Class present\' (for every class over every audio file)"] = ratios

        with open(os.path.join("plots/bird_dataset/", f"Bird_stats.yaml"), "w") as fp:
            yaml.dump(bird_data_dict, fp)
