import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Schritt 1: Einlesen des Datensatzes
csv_file = "Aufgabe 4/Praktikum4_Datensatz.csv"
data = []

with open(csv_file, "r") as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Header-Zeile überspringen
    for row in csv_reader:
        data.append(row)

# Schritt 2: Train/Val/Test Split
total_samples = len(data)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size : train_size + val_size]
test_data = data[train_size + val_size :]

# Schritt 3: Preprocessing
X_train = np.array([row[:-1] for row in train_data], dtype=float)
y_train = np.array([row[-1] for row in train_data])

X_val = np.array([row[:-1] for row in val_data], dtype=float)
y_val = np.array([row[-1] for row in val_data])

X_test = np.array([row[:-1] for row in test_data], dtype=float)
y_test = np.array([row[-1] for row in test_data])

# Standardisierung der kontinuierlichen Features
scaler = StandardScaler()
X_train[:, [0, 2, 4]] = scaler.fit_transform(X_train[:, [0, 2, 4]])
X_val[:, [0, 2, 4]] = scaler.transform(X_val[:, [0, 2, 4]])
X_test[:, [0, 2, 4]] = scaler.transform(X_test[:, [0, 2, 4]])

# One-Hot-Encoding der kategorialen Features
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_train_categorical = encoder.fit_transform(X_train[:, [1, 3]])
X_val_categorical = encoder.transform(X_val[:, [1, 3]])
X_test_categorical = encoder.transform(X_test[:, [1, 3]])

# Zusammenführen der Features
X_train = np.hstack((X_train[:, [0, 2, 4]], X_train_categorical))
X_val = np.hstack((X_val[:, [0, 2, 4]], X_val_categorical))
X_test = np.hstack((X_test[:, [0, 2, 4]], X_test_categorical))

# Konvertieren zu PyTorch-Tensoren
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train).unsqueeze(1)  # Unsere Ausgabe ist eine Spalte
X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val).unsqueeze(1)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test).unsqueeze(1)


# Schritt 4: Erstellung und Training des neuronalen Netzwerks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=X_train.shape[1], out_features=64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = Net()

# Optimierer und Loss-Funktion
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss für binäre Klassifikation

# Training des Modells
num_epochs = 10
batch_size = 64

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Evaluation auf Trainings- und Validierungsdaten
model.eval()
with torch.no_grad():
    train_preds = (model(X_train) > 0.5).float()
    val_preds = (model(X_val) > 0.5).float()

train_accuracy = accuracy_score(y_train, train_preds)
val_accuracy = accuracy_score(y_val, val_preds)
train_precision = precision_score(y_train, train_preds)
val_precision = precision_score(y_val, val_preds)
train_recall = recall_score(y_train, train_preds)
val_recall = recall_score(y_val, val_preds)

print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
print(
    f"Train Precision: {train_precision:.4f}, Validation Precision: {val_precision:.4f}"
)
print(f"Train Recall: {train_recall:.4f}, Validation Recall: {val_recall:.4f}")

# Du kannst nun Schritt 4 wiederholen, indem du verschiedene Hyperparameter und Architekturen ausprobierst,
# um die Leistung des Modells zu verbessern. Wenn du mit dem Modell zufrieden bist, bewerte es auf dem Testdatensatz.

# Schließlich bewerte das beste Modell auf dem Testdatensatz
model.eval()
with torch.no_grad():
    test_preds = (model(X_test) > 0.5).float()

test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
