import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from ColumnNames import ColumnNames

# define size of train, test and validation ratios
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

input_size = 9
hidden_size = 4
output_size = 1

num_epochs = 100


# https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch/63271002#63271002
def calc_accuracy(mdl: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> (float, float, float):
    """
    Get the accuracy with respect to the most likely label

    :param mdl:
    :param X:
    :param Y:
    :return:
    """
    # get the scores for each class (or logits)
    y_logits = mdl(X)  # unnormalized probs
    # return the values & indices with the largest value in the dimension where the scores for each class is
    # get the scores with largest values & their corresponding idx (so the class that is most likely)
    max_scores, max_idx_class = mdl(X).max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    # usually 0th coordinate is batch size
    _n = X.size(0)
    assert(_n == max_idx_class.size(0))
    # calulate acc (note .item() to do float division)
    acc = (max_idx_class == Y).sum().item() / _n

    # Calculate True Positives, False Positives, False Negatives
    true_positives = ((max_idx_class == Y) & (Y == 1)).sum().item()
    true_negatives = ((max_idx_class == Y) & (Y == 0)).sum().item()
    false_positives = ((max_idx_class != Y) & (Y == 0)).sum().item()
    false_negatives = ((max_idx_class != Y) & (Y == 1)).sum().item()

    # Calculate Precision and Recall
    prec = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
    rec = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0

    return acc, prec, rec


def check_for_early_stop(epoch_loss_val_plot) -> bool:
    if len(epoch_loss_val_plot) <= 5:
        return False

    if epoch_loss_val_plot[-5] < epoch_loss_val_plot[-1]:
        return True

    return False


if __name__ == '__main__':
    X = pd.read_csv('../Praktikum4_Datensatz.csv')

    x, y = (
        X[[
            ColumnNames.Grundstuecksgroesse.value,
            ColumnNames.Stadt.value,
            ColumnNames.Hausgroesse.value,
            ColumnNames.Kriminalitaetsindex.value,
            ColumnNames.Baujahr.value]
        ], X[ColumnNames.Klasse.value]
    )

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    numeric_features = [ColumnNames.Grundstuecksgroesse.value, ColumnNames.Hausgroesse.value, ColumnNames.Baujahr.value]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = [ColumnNames.Stadt.value, ColumnNames.Kriminalitaetsindex.value]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.fit_transform(X_test, y_test)
    X_val_transformed = preprocessor.fit_transform(X_val, y_val)

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(y_train.values.reshape(-1, 1))
    y_train_transformed = one_hot_encoder.transform(y_train.values.reshape(-1, 1)).toarray()[:, 1]

    one_hot_encoder.fit(y_test.values.reshape(-1, 1))
    y_test_transformed = one_hot_encoder.transform(y_test.values.reshape(-1, 1)).toarray()[:, 1]

    one_hot_encoder.fit(y_val.values.reshape(-1, 1))
    y_val_transformed = one_hot_encoder.transform(y_val.values.reshape(-1, 1)).toarray()[:, 1]

    # Überführen in Tensor Datensätze
    tensor_X_train = torch.tensor(X_train_transformed, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train_transformed, dtype=torch.float32)

    tensor_X_test = torch.tensor(X_test_transformed, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test_transformed, dtype=torch.float32)

    tensor_X_val = torch.tensor(X_val_transformed, dtype=torch.float32)
    tensor_y_val = torch.tensor(y_val_transformed, dtype=torch.float32)

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    val_dataset = TensorDataset(tensor_X_val, tensor_y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid()
    )

    loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_loss_train_plot = []
    epoch_loss_val_plot = []

    for n in range(num_epochs):
        epoch_loss_train = 0
        model.train()
        for X, y in train_dataloader:
            y_pred = model(X)
            loss = loss_function(torch.squeeze(y_pred), y)  # Soll-Ist-Wert Vergleich mit Fehlerfunktion
            epoch_loss_train += loss.item()
            optimizer.zero_grad()  # Gradient ggf. vom vorherigen Durchlauf auf 0 setzen
            loss.backward()  # Backpropagation
            optimizer.step()  # Gradientenschritt
        epoch_loss_train_plot.append(epoch_loss_train)

        epoch_loss_val = 0
        model.eval()
        for X, y in val_dataloader:
            y_pred = model(X)
            loss = loss_function(torch.squeeze(y_pred), y)
            epoch_loss_val += loss.item()
        epoch_loss_val_plot.append(epoch_loss_val)

        if check_for_early_stop(epoch_loss_val_plot):
            break

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    plt.title("Loss")
    plt.plot(epoch_loss_train_plot, color='red', label='train')
    plt.plot(epoch_loss_val_plot, color='blue', label='val')
    plt.legend()
    plt.show()

    accuracy_train = 0
    precision_train = 0
    recall_train = 0
    counter_train = 0
    for X, y in train_dataloader:
        a, p, r = calc_accuracy(model, X, y)
        accuracy_train += a
        precision_train += p
        recall_train += r
        counter_train += 1

    accuracy_val = 0
    precision_val = 0
    recall_val = 0
    counter_val = 0
    for X, y in val_dataloader:
        a, p, r = calc_accuracy(model, X, y)
        accuracy_val += a
        precision_val += p
        recall_val += r
        counter_val += 1

    print('Accuracy of train data: {}'.format(accuracy_train / counter_train))
    print('Accuracy of val data: {}'.format(accuracy_val / counter_val))

    print('Precision of train data: {}'.format(precision_train / counter_train))
    print('Precision of val data: {}'.format(precision_val / counter_val))

    print('Recall of train data: {}'.format(recall_train / counter_train))
    print('Recall of val data: {}'.format(recall_val / counter_val))

    accuracy_test = 0
    precision_test = 0
    recall_test = 0
    counter_test = 0
    for X, y in test_dataloader:
        a, p, r = calc_accuracy(model, X, y)
        accuracy_test += a
        precision_test += p
        recall_test += r
        counter_test += 1

    print('Accuracy of test data: {}'.format(accuracy_test / counter_test))
    print('Precision of test data: {}'.format(precision_test / counter_test))
    print('Recall of test data: {}'.format(recall_test / counter_test))

