import torch
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import numpy as np

def evaluate(model, device, data_loader, id2cls, decision_threshold=0.5, apply_sigmoid=True):
    model.to(device)
    model.eval()
    num_classes = len(id2cls.keys())

    y = []
    y_hat = []
    for data in data_loader:
        features, labels = data
        if type(features) is list:
            features = [feature.to(device) for feature in features]
        else:
            features = features.to(device)
        batch_pred = model(features)
        if apply_sigmoid:
            batch_pred = torch.sigmoid(batch_pred)
        batch_pred = torch.where(batch_pred > decision_threshold, 1, 0).to('cpu')

        y.append(labels)
        y_hat.append(batch_pred)

    results = {}
    y = np.concatenate(y, axis=0).astype(np.int64)
    y = np.reshape(y, (-1, num_classes))
    y_hat = np.concatenate(y_hat, axis=0).astype(np.int64)
    y_hat = np.reshape(y_hat, (-1, num_classes))

    f1s = f1_score(y, y_hat, average=None).tolist()
    precisions = precision_score(y, y_hat, average=None).tolist()
    recalls = recall_score(y, y_hat, average=None).tolist()
    if num_classes == 1:
        class_results = {list(id2cls.values())[0]: {'f1': f1s[1], 'precision': precisions[1], 'recall': recalls[1]}}
    else:
        class_results = {cls: {'f1': f1s[id], 'precision': precisions[id], 'recall': recalls[id]} for id, cls in id2cls.items()}

    y = y.flatten()
    y_hat = y_hat.flatten()
    results['recall'] = float(recall_score(y, y_hat))
    results['precision'] = float(precision_score(y, y_hat))
    results['f1'] = float(f1_score(y, y_hat))
    results['accuracy'] = float(accuracy_score(y, y_hat))

    return results, class_results

def get_prediction_from_raw_output(raw_prediction, id2cls, audio_duration, block_length, file_name, decision_threshold=0.5, apply_sigmoid=False):
    if apply_sigmoid:
        raw_prediction = torch.sigmoid(raw_prediction)
    raw_prediction = torch.where(raw_prediction > decision_threshold, 1, 0).to('cpu')

    # convert block_length to ms
    block_length = float(block_length) / 1000

    # cutoff prediction at end of audio file
    cutoff = int(audio_duration / block_length) + 1

    found_events = []
    for idx, cls in id2cls.items():
        if len(raw_prediction.shape) == 1:
            raw_prediction = torch.unsqueeze(raw_prediction, 1)
        cls_slice = raw_prediction[:cutoff, idx].detach().clone()

        array = cls_slice.numpy()

        # Find the start and end points of each block of 1s
        blocks = np.where(np.diff(np.concatenate(([0], array, [0]))))[0].reshape(-1, 2)
        blocks = [(start, end) for start, end in blocks]

        # convert block to seconds and append to found events
        for block in blocks:
            start, end = block
            start = start * block_length
            end = end * block_length

            found_events.append((file_name, start, end, cls))

    return found_events


def evaluate_single_label(model, device, data_loader, id2cls, decision_threshold=0.5):
    # seperate evaluation method for Single label per file BirdSED
    model.to(device)
    model.eval()
    num_classes = len(id2cls.keys())

    y = []
    y_hat = []
    cls_y = []
    cls_y_hat = []
    for data in data_loader:
        features, labels = data
        sed_labels, cls_labels = labels
        if type(features) is list:
            features = [feature.to(device) for feature in features]
        else:
            features = features.to(device)
        batch_sed_pred, batch_cls_pred = model(features)
        batch_sed_pred = torch.where(batch_sed_pred > decision_threshold, 1, 0).to('cpu')

        y.append(sed_labels)
        y_hat.append(batch_sed_pred)

        batch_cls_pred = torch.argmax(batch_cls_pred, dim=1).to('cpu')
        cls_labels = torch.argmax(cls_labels, dim=1)

        cls_y.append(cls_labels)
        cls_y_hat.append(batch_cls_pred)

    results = {}
    y = np.concatenate(y, axis=0).astype(np.int64)
    y = np.reshape(y, (-1, num_classes))
    y_hat = np.concatenate(y_hat, axis=0).astype(np.int64)
    y_hat = np.reshape(y_hat, (-1, num_classes))

    f1s = f1_score(y, y_hat, average=None).tolist()
    precisions = precision_score(y, y_hat, average=None).tolist()
    recalls = recall_score(y, y_hat, average=None).tolist()
    if num_classes == 1:
        class_results = {list(id2cls.values())[0]: {'f1': f1s[1], 'precision': precisions[1], 'recall': recalls[1]}}
    else:
        class_results = {cls: {'f1': f1s[id], 'precision': precisions[id], 'recall': recalls[id]} for id, cls in id2cls.items()}

    y = y.flatten()
    y_hat = y_hat.flatten()
    results['SED'] = {}
    results['SED']['recall'] = float(recall_score(y, y_hat))
    results['SED']['precision'] = float(precision_score(y, y_hat))
    results['SED']['f1'] = float(f1_score(y, y_hat))
    results['SED']['accuracy'] = float(accuracy_score(y, y_hat))

    cls_y = np.concatenate(cls_y, axis=0).astype(np.int64)
    cls_y_hat = np.concatenate(cls_y_hat, axis=0).astype(np.int64)

    results['Classification'] = {}
    results['Classification']['recall'] = float(recall_score(cls_y, cls_y_hat, average='micro'))
    results['Classification']['precision'] = float(precision_score(cls_y, cls_y_hat, average='micro'))
    results['Classification']['f1'] = float(f1_score(cls_y, cls_y_hat, average='micro'))
    results['Classification']['accuracy'] = float(accuracy_score(cls_y, cls_y_hat))

    return results, class_results
