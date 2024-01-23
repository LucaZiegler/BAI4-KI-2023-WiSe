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
from Metrics import calculateMetrics

# define size of train, test and validation ratios
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

input_size = 9
hidden_size = 200
output_size = 1

num_epochs = 200
learning_rate = 0.001


def check_for_early_stop(_epoch_loss_val_plot) -> bool:
    if len(_epoch_loss_val_plot) <= 5:
        return False

    if _epoch_loss_val_plot[-5] < _epoch_loss_val_plot[-1]:
        return True

    return False


def visualize(_epoch_loss_train_plot, _epoch_loss_val_plot):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.title("Loss")
    plt.plot(_epoch_loss_train_plot, color='red', label='train')
    plt.plot(_epoch_loss_val_plot, color='blue', label='val')
    plt.legend()
    plt.show()


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

    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=50, shuffle=False)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        # nn.Tanh(),
        # nn.Linear(hidden_size, hidden_size),
        # nn.Tanh(),
        nn.Tanh(),
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, 30),
        nn.Tanh(),
        nn.BatchNorm1d(30),
        nn.Linear(30, 15),
        nn.Tanh(),
        nn.BatchNorm1d(15),
        nn.Linear(15, 5),
        nn.Tanh(),
        nn.BatchNorm1d(5),
        nn.Linear(5, output_size),
        #nn.LazyLinear(output_size),
        nn.Sigmoid()
    )

    loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate)

    epoch_loss_train_plot = []
    epoch_loss_val_plot = []

    for n in range(num_epochs):
        epoch_loss_train = 0
        model.train()
        epoch_train_pred = []
        epoch_train_data = []
        epoch_val_data = []
        epoch_val_pred = []

        for X, y in train_dataloader:
            y_pred = model(X)
            epoch_train_data.append(y)
            epoch_train_pred.append(y_pred)
            loss = loss_function(torch.squeeze(y_pred), y)  # Soll-Ist-Wert Vergleich mit Fehlerfunktion
            epoch_loss_train += loss.item()
            optimizer.zero_grad()  # Gradient ggf. vom vorherigen Durchlauf auf 0 setzen
            loss.backward()  # Backpropagation
            optimizer.step()  # Gradientenschritt

        epoch_loss_train_plot.append(epoch_loss_train/len(train_dataset))

        epoch_loss_val = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_dataloader:
                y_pred = model(X)
                epoch_val_data.append(y)
                epoch_val_pred.append(y_pred)
                loss = loss_function(torch.squeeze(y_pred), y)
                epoch_loss_val += loss.item()

        epoch_loss_val_plot.append(epoch_loss_val / len(val_dataset))

        if check_for_early_stop(epoch_loss_val_plot):
            break

    visualize(epoch_loss_train_plot, epoch_loss_val_plot)

    calculateMetrics(train_dataloader, val_dataloader, test_dataloader, model)

# Für welche, eigentlich
# immer genutzte und recht simple, Regularisierungsmethode sind diese Werte
# notwendig?
# → Early Stopping

# 4. Warum? Wie nennt sich das in den obigen vier Schritten beschriebene Vorgehen?
# → Kreuzvalidierung (2-fach)
# → Dieser Prozess wird oft als Hyperparameter-Optimierung oder Modellfine-tuning bezeichnet.

# Der Evaluierungsmodus ist insbesondere bei Dropout und Batch Normalization wichtig (Warum?).
# → Im Evaluierungsmodus wird Dropout deaktiviert, da es keinen Sinn ergibt einzelne Neuronen zum Validieren
#       auszuschalten.

# → Während des Trainingsmodus werden die Mittelwerte und Standardabweichungen jeder Mini-Batch berechnet und
#       verwendet, um die Normalisierung durchzuführen.
#       Im Evaluierungsmodus möchte man jedoch eine konsistente Normalisierung basierend auf den gesamten Datensatz
#       erhalten, nicht nur auf einer Mini-Batch. Daher werden während der Evaluierung die Mittelwerte und
#       Standardabweichungen über den gesamten Datensatz berechnet und für die Normalisierung verwendet.

# BatchNormalization und Dropout sollten nicht gleichzeitig verwendet werden !
# Dropout zerstört die Batch-Statistik.