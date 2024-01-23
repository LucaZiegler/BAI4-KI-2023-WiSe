import torch


def calculateMetrics(train_dataloader, val_dataloader, test_dataloader, model):
    accuracy_train, precision_train, recall_train = calc(train_dataloader, model)
    accuracy_val, precision_val, recall_val = calc(val_dataloader, model)
    accuracy_test, precision_test, recall_test = calc(test_dataloader, model)

    print('Accuracy of train data: {}'.format(accuracy_train))
    print('Precision of train data: {}'.format(precision_train))
    print('Recall of train data: {}'.format(recall_train))
    print('Accuracy of val data: {}'.format(accuracy_val))
    print('Precision of val data: {}'.format(precision_val))
    print('Recall of val data: {}'.format(recall_val))
    print('Accuracy of test data: {}'.format(accuracy_test))
    print('Precision of test data: {}'.format(precision_test))
    print('Recall of test data: {}'.format(recall_test))


def calc(dataloader, model):
    epoch_pred = []
    epoch_data = []

    for X, y in dataloader:
        y_pred = model(X)
        epoch_data.append(y)
        epoch_pred.append(y_pred)

    a, p, r = acc_prec_rec_score(torch.cat(epoch_pred), torch.cat(epoch_data))

    return a, p, r


def convert_to_binary(predictions, threshold=0.5):
    if predictions > threshold:
        return 1
    else:
        return 0


def acc_prec_rec_score(y_pred, y):
    acc = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for pred in range(len(y_pred)):
        binary_pred = convert_to_binary(y_pred[pred].item())
        if binary_pred == y[pred].item():
            acc += 1
        if binary_pred == 1:
            if y[pred].item() == 1:
                true_positives += 1
            else:
                false_positives += 1
        if binary_pred == 0:
            if y[pred].item() == 0:
                true_negatives += 1
            else:
                false_negatives += 1

    prec = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
    rec = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0

    return acc / len(y), prec, rec