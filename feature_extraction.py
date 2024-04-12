import glob
import os
import librosa
import numpy as np

from utils import get_wav_duration

sampling_rates = {
    'TUT': 44100
}

audio_file_dirs = {
    'TUT': "dataset/TUT/audio/street/*"
}


def extract_features(dataset, clip_length, n_fft, n_mels, hop_length, win_length):
    audio_files = glob.glob(audio_file_dirs[dataset])
    sampling_rate = sampling_rates[dataset]

    clip_length = 0.001 * clip_length # ms to s

    features = {}
    for audio_file in audio_files:
        audio_file = audio_file.replace("\\", "/")
        file_name = "/".join(audio_file.split('/')[-3:])
        audio_length = get_wav_duration(audio_file)

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