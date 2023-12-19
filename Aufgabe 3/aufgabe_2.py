import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


df = pd.read_csv("Aufgabe 3\Praktikum3_Datensatz.csv")

# df.info()

# https://stackoverflow.com/a/49271663/4503373
le = LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass
# https://stackoverflow.com/a/49271663/4503373


# Train/Test Split
train_size = 0.8
df_train, df_test = train_test_split(df, test_size=1 - train_size, random_state=42)


# Definiere Spalten für kontinuierliche und diskrete Variablen
continuous_cols = ["grundstuecksgroesse", "hausgroesse", "kriminalitaetsindex"]
discrete_cols = ["stadt", "baujahr"]
# Nominal: Stadt
# Intervall: Baujahr
# Verhältnisskala: Grundstücksgrösse, Hausgrösse, Kriminalitätsindex

# Spalten-Transformer für die Preprocessing-Schritte
# Wichtig für konsistente Daten
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_cols),
        ("cat", OneHotEncoder(), discrete_cols),
    ]
)

# Wende Preprocessing auf Trainingsdaten an
X_train = preprocessor.fit_transform(df_train.drop("klasse", axis=1))
y_train = df_train["klasse"]

# Wende Preprocessing auf Testdaten an (mit den auf den Trainingsdaten gelernten Parametern)
X_test = preprocessor.transform(df_test.drop("klasse", axis=1))
y_test = df_test["klasse"]

# Training der logistischen Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Evaluation
# Vorhersagen auf dem Testdatensatz
y_pred = logreg.predict(X_test)

# Berechne Metriken
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# score = logreg.score(X_train,y_train)

# Ausgabe der Metriken
print(f"Accuracy: {accuracy}")  # Genauigkeit des Anteil der korrekt klassifierten Daten
print(f"Precision: {precision}")  # Genauigkeit der positiven Vorhersagen
print(f"Recall: {recall}")  # Sensitivität oder Trefferquote

# print(f"Accuracy - : {score}") # Percentage that are true

# Mit Penality ist L1/L2 Regularisierung einstellen
