import torch
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import numpy as np

def evaluate(model, device, data_loader, id2cls, decision_threshold=0.5):
    model.to(device)
    model.eval()

    y = []
    y_hat = []
    for data in data_loader:
        features, labels = data
        if type(features) is list:
            features = [feature.to(device) for feature in features]
        else:
            features = features.to(device)
        batch_pred = torch.sigmoid(model(features))
        batch_pred = torch.where(batch_pred > decision_threshold, 1, 0).to('cpu')

        y.append(labels)
        y_hat.append(batch_pred)

    results = {}
    y = np.concatenate(y, axis=0).astype(np.int64)
    y = np.reshape(y, (-1, y.shape[-1]))
    y_hat = np.concatenate(y_hat, axis=0).astype(np.int64)
    y_hat = np.reshape(y_hat, (-1, y_hat.shape[-1]))

    f1s = f1_score(y, y_hat, average=None).tolist()
    precisions = precision_score(y, y_hat, average=None).tolist()
    recalls = recall_score(y, y_hat, average=None).tolist()
    class_results = {cls: {'f1': f1s[id], 'precision': precisions[id], 'recall': recalls[id]} for id, cls in id2cls.items()}

    y = y.flatten()
    y_hat = y_hat.flatten()
    results['recall'] = float(recall_score(y, y_hat))
    results['precision'] = float(precision_score(y, y_hat))
    results['f1'] = float(f1_score(y, y_hat))
    results['accuracy'] = float(accuracy_score(y, y_hat))

    return results, class_results