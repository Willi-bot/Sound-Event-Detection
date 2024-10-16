import glob
import os
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTModel

sampling_rates = {
    'TUT': 44100,
    'desed_2022': 16000,
    'BirdSED': 16000
}


def extract_mel_spectrograms(audio_files, dataset_location, dataset, clip_length, n_fft, n_mels, hop_length, win_length, save_location=None, get_ast_features=False):
    sampling_rate = sampling_rates[dataset]
    clip_length = 0.001 * clip_length # ms to s

    if get_ast_features:
        feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        pooling = torch.nn.AdaptiveAvgPool2d((1000, 768))
    else:
        feature_extractor = None
        ast_model = None
        pooling = None

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

            if not get_ast_features:
                mel_spec = librosa.feature.melspectrogram(y=audio_clip, sr=sampling_rate, n_fft=n_fft, n_mels=n_mels,
                                                          hop_length=hop_length, win_length=win_length)

                # switch frequency and time axis
                shape = mel_spec.shape
                feature = np.reshape(mel_spec, (shape[1], shape[0]))
            else:
                if sampling_rate != 16000:
                    audio_clip = librosa.resample(audio_clip, orig_sr=sampling_rate, target_sr=16000)
                feature = feature_extractor(audio_clip, sampling_rate=16000, return_tensors='pt')
                feature = ast_model(**feature)['last_hidden_state']
                # i need the sequence length to be 1000
                feature = pooling(feature)
                feature = feature.squeeze().detach().cpu().numpy()

            if save_location is not None:
                audio_file_no_slashes = audio_file.replace('/', '_slash_')
                mel_file_path = save_location + '/' + audio_file_no_slashes + '_' + str(i) + '.npy'
                filename = audio_file_no_slashes + '_' + str(i) + '.npy'
                np.save(mel_file_path, feature)

                features[filename] = mel_file_path
            else:
                features[(i, audio_file)] = feature

    return features


def get_features(name, dataset_location, dataset, fold, clip_length, n_fft, n_mels, hop_length, win_length, audio_file_folder=None, audio_files=None, get_ast_features=False):
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

        features = extract_mel_spectrograms(audio_files, dataset_location, dataset, clip_length, n_fft, n_mels, hop_length, win_length, save_location=dir_path, get_ast_features=get_ast_features)
        print("Done!")

    return features