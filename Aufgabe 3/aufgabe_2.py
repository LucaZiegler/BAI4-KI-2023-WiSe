import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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


# Schritt 2: Train/Test Split
train_size = 0.8
df_train, df_test = train_test_split(df, test_size=1 - train_size, random_state=42)


# Schritt 3: Preprocessing
# Definiere Spalten für kontinuierliche und diskrete Variablen
continuous_cols = ["grundstuecksgroesse", "hausgroesse", "kriminalitaetsindex"]
discrete_cols = ["stadt", "baujahr"]

# Spalten-Transformer für die Preprocessing-Schritte
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

# Schritt 4: Training der logistischen Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print(f"Accuracy - : {logreg.score(X_train,y_train)}") # Percentage that are true
