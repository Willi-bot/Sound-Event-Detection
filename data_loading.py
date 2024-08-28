import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking


class AudioClipDataset(Dataset):
    def __init__(self, features, labels, use_specaug=False):
        self.features = features
        self.labels = labels

        if use_specaug:
            self.spec_aug = torch.nn.Sequential(
                TimeStretch(0.8, fixed_rate=True),
                FrequencyMasking(freq_mask_param=80),
                TimeMasking(time_mask_param=80),
            )
        else:
            self.spec_aug = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = np.load(self.features[idx])
        labels = np.load(self.labels[idx])

        features = torch.from_numpy(features)

        if self.spec_aug is not None:
            features = self.spec_aug(features)


        if type(labels) is tuple:
            labels = (torch.from_numpy(labels[0]).float(), torch.from_numpy(labels[1]).float())
        else:
            labels = torch.from_numpy(labels).float()

        return features.float(), labels


def get_dataloader(features, labels, batch_size, shuffle=False, drop_last=False, use_specaug=False, num_workers=0):
    # turn dict to list
    features_list, labels_list = [], []
    for key in features.keys():
        if key in labels.keys():
            features_list.append(features[key])
            labels_list.append(labels[key])

    dataset = AudioClipDataset(features_list, labels_list, use_specaug=use_specaug)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=True)

    return dataloader
