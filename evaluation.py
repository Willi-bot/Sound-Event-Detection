import torch
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np

def evaluate(model, device, data_loader):
    model.to(device)
    model.eval()

    y = []
    y_hat = []
    for features, labels in data_loader:
        features = features.to(device)
        batch_pred = torch.sigmoid(model(features))
        batch_pred = torch.where(batch_pred > 0.5, 1, 0).to('cpu')

        y.append(labels)
        y_hat.append(batch_pred)

    y = np.concatenate(y, axis=0).flatten()
    y_hat = np.concatenate(y_hat, axis=0).flatten()
    recall = recall_score(y, y_hat)
    precision = precision_score(y, y_hat)
    f1 = f1_score(y, y_hat)

    return f1, recall, precision