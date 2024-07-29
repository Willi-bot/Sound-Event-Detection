import glob
import os
import librosa
import numpy as np
from tqdm import tqdm

sampling_rates = {
    'TUT': 44100,
    'desed_2022': 16000,
    'BirdSED': 16000
}


def extract_mel_spectrograms(audio_files, dataset_location, dataset, clip_length, n_fft, n_mels, hop_length, win_length, save_location):
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

            audio_file_no_slashes = audio_file.replace('/', '_slash_')
            mel_file_path = save_location + '/' + audio_file_no_slashes + '_' + str(i) + '.npy'
            np.save(mel_file_path, mel_spec)

            features[(i, audio_file)] = mel_file_path

    return features


def get_features(name, dataset_location, dataset, fold, clip_length, n_fft, n_mels, hop_length, win_length, audio_file_folder=None, audio_files=None):
    dir_name = name + f'_{fold}_{clip_length}_{n_fft}_{n_mels}_{hop_length}_{win_length}'
    dir_path = dataset_location + dataset + '/' + dir_name

    if os.path.isdir(dir_path):
        print('getting feature dict from ' + dir_path + "...")
        # get features dict
        features = {}
        for filename in glob.glob(dir_path + '/*.npy', ):
            key_name = filename.replace('\\', '/').split('/')[-1]
            features[key_name] = filename
    else:
        os.makedirs(dir_path, exist_ok=True)

        if audio_file_folder is not None:
            audio_path = dataset_location + dataset + '/' + audio_file_folder
            print('extracting features from ' + audio_path + "...")
            audio_files = glob.glob(audio_path)
            audio_files = [file.replace(dataset_location + dataset + '/', '') for file in audio_files]
        elif audio_files is not None:
            print('extracting and saving features from given audio files...')
        else:
            # TODO
            pass

        features = extract_mel_spectrograms(audio_files, dataset_location, dataset, clip_length, n_fft, n_mels, hop_length, win_length, dir_path)
        print("Done!")

    return features