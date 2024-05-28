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
        features = torch.from_numpy(self.features[idx])

        if self.spec_aug is not None:
            features = self.spec_aug(features)

        return features.float(), torch.from_numpy(self.labels[idx]).float()


def get_dataloader(features, labels, batch_size, shuffle=False, drop_last=False, use_specaug=False):
    dataset = AudioClipDataset(features, labels, use_specaug=use_specaug)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader
