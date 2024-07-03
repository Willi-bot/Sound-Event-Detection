import glob
import os
import librosa
import numpy as np
from tqdm import tqdm

sampling_rates = {
    'TUT': 44100,
    'desed_2022': 16000,
    'bird_dataset': 16000
}


def extract_mel_spectrograms(audio_files, dataset_location, dataset, clip_length, n_fft, n_mels, hop_length, win_length):
    sampling_rate = sampling_rates[dataset]
    clip_length = 0.001 * clip_length # ms to s

    features = {}
    for audio_file in tqdm(audio_files, disable=len(audio_files) == 1):
        file_path = dataset_location + '/' + dataset + '/' + audio_file
        audio_file = audio_file.replace('\\', '/')
        audio_length = librosa.get_duration(path=file_path)

        for i in range(int(np.ceil(audio_length / clip_length))):
            offset = i * clip_length
            audio_clip, _ = librosa.load(file_path, sr=sampling_rate, offset=offset,
                                         duration=clip_length)

            # if audio clip too short, pad with zeros
            if len(audio_clip) < sampling_rate * clip_length:
                audio_clip = np.append(audio_clip, np.zeros(int(sampling_rate * clip_length) - len(audio_clip)))

            mel_spec = librosa.feature.melspectrogram(y=audio_clip, sr=sampling_rate, n_fft=n_fft, n_mels=n_mels,
                                                      hop_length=hop_length, win_length=win_length)

            # switch frequency and time axis
            shape = mel_spec.shape
            mel_spec = np.reshape(mel_spec, (shape[1], shape[0]))

            features[(i, audio_file)] = mel_spec

    return features


def get_features(name, dataset_location, dataset, fold, clip_length, n_fft, n_mels, hop_length, win_length, audio_file_folder=None, audio_files=None):
    file_name = name + f'_{fold}_{clip_length}_{n_fft}_{n_mels}_{hop_length}_{win_length}.npy'
    file_path = dataset_location + dataset + '/' + file_name

    if os.path.isfile(file_path):
        print('loading features from ' + file_path + "...")
        features = np.load(file_path, allow_pickle=True).item()
    else:
        if audio_file_folder is not None:
            audio_path = dataset_location + dataset + '/' + audio_file_folder
            print('extracting features from ' + audio_path + "...")
            audio_files = glob.glob(audio_path)
            audio_files = [file.replace(dataset_location + dataset + '/', '') for file in audio_files]
        elif audio_files is not None:
            print('extracting features from given audio files...')
        else:
            # TODO
            pass

        features = extract_mel_spectrograms(audio_files, dataset_location, dataset, clip_length, n_fft, n_mels, hop_length, win_length)
        print("saving features...")
        np.save(file_path, features)

    return features