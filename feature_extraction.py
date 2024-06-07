import glob
import os
import librosa
import numpy as np
import torch
from tqdm import tqdm

from utils import get_wav_duration

sampling_rates = {
    'TUT': 44100,
    'desed_2022': 16000
}

audio_file_dirs = {
    'TUT': "dataset/TUT/audio/street/*.wav",
    'desed_2022': "dataset/desed_2022/audio/**/*.wav"
}

desed_audiofolder2type = {
    'strong_label_real_16k': 'strong',
    'synthetic21_train/soundscapes_16k': 'synthetic',
    'synthetic21_validation/soundscapes_16k': 'synthetic',
    'validation_16k': 'strong'
}


def extract_mel_features(dataset, clip_length, n_fft, n_mels, hop_length, win_length):
    audio_files = glob.glob(audio_file_dirs[dataset], recursive=True)
    clip_length = 0.001 * clip_length # ms to s

    if dataset == 'TUT':
        features = get_tut_features(audio_files, clip_length, n_fft, n_mels, hop_length, win_length)
    elif dataset == 'desed_2022':
        not_strongly_labeled = ['unlabel_in_domain', 'weak']
        # remove weakly labeled data
        audio_files = [audio_file for audio_file in audio_files if all(not_strong not in audio_file for not_strong in not_strongly_labeled)]
        # only keep audio files with sampling rate = 16000
        audio_files = [audio_file for audio_file in audio_files if '_16k\\' in audio_file]
        features = get_desed_features(audio_files, clip_length, n_fft, n_mels, hop_length, win_length)
    else:
        features = None

    return features


def get_mfcc(mel_features, n_mfcc):
    mfccs = librosa.feature.mfcc(S=mel_features, n_mfcc=n_mfcc)

    return mfccs


def extract_mel_spectrograms(audio_files, dataset, clip_length, n_fft, n_mels, hop_length, win_length):
    sampling_rate = sampling_rates[dataset]
    clip_length = 0.001 * clip_length # ms to s

    features = {}
    for audio_file in tqdm(audio_files):
        audio_file = audio_file.replace("\\", "/")
        file_name = audio_file.split('/')[-1]
        audio_length = librosa.get_duration(path=audio_file)

        for i in range(int(np.ceil(audio_length / clip_length))):
            offset = i * clip_length
            audio_clip, _ = librosa.load(audio_file, sr=sampling_rate, offset=offset,
                                         duration=clip_length)

            # if audio clip too short, pad with zeros
            if len(audio_clip) < sampling_rate * clip_length:
                audio_clip = np.append(audio_clip, np.zeros(int(sampling_rate * clip_length) - len(audio_clip)))

            mel_spec = librosa.feature.melspectrogram(y=audio_clip, sr=sampling_rate, n_fft=n_fft, n_mels=n_mels,
                                                      hop_length=hop_length, win_length=win_length)

            # switch frequency and time axis
            shape = mel_spec.shape
            mel_spec = np.reshape(mel_spec, (shape[1], shape[0]))

            features[(i, file_name)] = mel_spec

    return features


def get_tut_features(audio_files, clip_length, n_fft, n_mels, hop_length, win_length):
    sampling_rate = sampling_rates['TUT']

    features = {}
    for audio_file in audio_files:
        audio_file = audio_file.replace("\\", "/")
        file_name = "/".join(audio_file.split('/')[-3:])
        audio_length = librosa.get_duration(path=audio_file)

        for i in range(int(audio_length // clip_length) + 1):
            offset = i * clip_length
            audio_clip, _ = librosa.load(audio_file, sr=sampling_rate, offset=offset,
                                         duration=clip_length)

            # if audio clip too short, pad with zeros
            if len(audio_clip) < sampling_rate * clip_length:
                audio_clip = np.append(audio_clip, np.zeros(int(sampling_rate * clip_length) - len(audio_clip)))

            mel_spec = librosa.feature.melspectrogram(y=audio_clip, sr=sampling_rate, n_fft=n_fft, n_mels=n_mels,
                                                      hop_length=hop_length, win_length=win_length)

            # switch frequency and time axis
            shape = mel_spec.shape
            mel_spec = np.reshape(mel_spec, (shape[1], shape[0]))

            features[(i, file_name)] = mel_spec

    return features


def get_desed_features(audio_files, clip_length, n_fft, n_mels, hop_length, win_length):
    sampling_rate = sampling_rates['desed_2022']

    features = {}
    for audio_file in audio_files:
        audio_file = audio_file.replace("\\", "/")
        if not 'synthetic' in audio_file:
            file_name = audio_file.split("/")[-3:]
            file_name = "/".join([file_name[0], desed_audiofolder2type[file_name[1]], file_name[2]])
        else:
            file_name = audio_file.split("/")[-4:]
            file_name = [file_name[0], desed_audiofolder2type[file_name[1] + "/" + file_name[2]], file_name[3]]
            file_name = "/".join(file_name)

        audio_clip, _ = librosa.load(audio_file, sr=sampling_rate)

        # if audio clip too short, pad with zeros
        if len(audio_clip) < sampling_rate * clip_length:
            audio_clip = np.append(audio_clip, np.zeros(int(sampling_rate * clip_length) - len(audio_clip)))

        mel_spec = librosa.feature.melspectrogram(y=audio_clip, sr=sampling_rate, n_fft=n_fft, n_mels=n_mels,
                                                  hop_length=hop_length, win_length=win_length)

        # switch frequency and time axis
        shape = mel_spec.shape
        mel_spec = np.reshape(mel_spec, (shape[1], shape[0]))

        features[file_name] = mel_spec

    return features


def extract_mel_features_single_file(audio_file, dataset, clip_length, n_fft, n_mels, hop_length, win_length):
    clip_length = 0.001 * clip_length  # ms to s
    if dataset == 'TUT':
        features = get_tut_features([audio_file], clip_length, n_fft, n_mels, hop_length, win_length)
    elif dataset == 'desed_2022':
        features = get_desed_features([audio_file], clip_length, n_fft, n_mels, hop_length, win_length)
    else:
        features = None

    return features


def get_features(name, dataset_location, dataset, clip_length, n_fft, n_mels, hop_length, win_length, audio_file_folder):
    file_name = name + f'_{clip_length}_{n_fft}_{n_mels}_{hop_length}_{win_length}.npy'
    file_path = dataset_location + '/' + dataset + '/' + file_name

    if os.path.isfile(file_path):
        print('loading features from ' + file_path + "...")
        features = np.load(file_path, allow_pickle=True).item()
    else:
        audio_path = dataset_location + '/' + dataset + '/'+ audio_file_folder
        print('extracting features from ' + audio_path + "...")
        audio_files = glob.glob(audio_path)

        features = extract_mel_spectrograms(audio_files, dataset, clip_length, n_fft, n_mels, hop_length, win_length)
        print("saving features...")
        np.save(file_path, features)

    return features