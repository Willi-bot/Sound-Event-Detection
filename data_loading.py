from torch.utils.data import Dataset, DataLoader


class AudioClipDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloader(features, labels, batch_size, shuffle=False, drop_last=False):
    dataset = AudioClipDataset(features, labels)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader