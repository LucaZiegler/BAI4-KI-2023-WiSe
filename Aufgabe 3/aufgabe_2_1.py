from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from enum import Enum


class ColumnNames(Enum):
    Grundstuecksgroesse = 'grundstuecksgroesse'
    Stadt = 'stadt'
    Hausgroesse = 'hausgroesse'
    Kriminalitaetsindex = 'kriminalitaetsindex'
    Baujahr = 'baujahr'
    Klasse = 'klasse'

data = pd.read_csv('Aufgabe 3\Praktikum3_Datensatz.csv')

x, y = (
    data[[
        ColumnNames.Grundstuecksgroesse.value,
        ColumnNames.Stadt.value,
        ColumnNames.Hausgroesse.value,
        ColumnNames.Kriminalitaetsindex.value,
        ColumnNames.Baujahr.value]
    ], data[ColumnNames.Klasse.value]
)

# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
numeric_features = [ColumnNames.Grundstuecksgroesse.value, ColumnNames.Hausgroesse.value, ColumnNames.Baujahr.value]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = [ColumnNames.Stadt.value, ColumnNames.Kriminalitaetsindex.value]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(solver='liblinear', penalty='l1'))]
)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)  # Fit the model according to the given training data.
accuracy_model = clf.score(X_test, y_test)

print("Accuracy by Pipeline: %.3f" % accuracy_model)  # Return the mean accuracy on the given test data and labels.

y_pred = clf.predict(X_test)

accuracy_metric = accuracy_score(y_test, y_pred)
precision_metric = precision_score(y_test, y_pred, average='weighted')
recall_metric = recall_score(y_test, y_pred, average='macro')

print("Accuracy by Metrics: %.3f\r\n" % accuracy_metric)
print("Precision: %.3f" % precision_metric)
print("Recall: %.3f" % recall_metric)

# Für die Testdaten, welche ja beim späteren Testdurchlauf auch
# standardisiert werden müssen, werden dann die (für die jeweiligen Spalten) auf dem
# Trainingsdatensatz ermittelten Mittel- und Varianzwerte für die Standardisierung verwendet (warum?).
# → Wir haben die Daten in 20 % Test- und 80 % Trainingsdaten eingeteilt, um sicherzustellen,
# → dass wir nicht alle Daten auswendig lernen.
# → Es werden nun die standardisierten Mittel- und Varianzwerte der Trainingsdaten auf die Testdaten angewendet
# → um zu testen, ob das gelernte auch auf andere Daten(mit gleicher Inputstruktur aber anderen Werten) anwendbar ist.

# Was ist mit dem Parameter 'penalty' einstellbar?
# → Hiermit kann die Art der Strafe, also die Regularisierung vorgegeben werden (L1, L2, None; abhängig vom Solver)
# → die auf die Kostenfunktion angewendet wird, um Overfitting zu vermeiden.

# Was ist grundsätzlich die jeweilige Aussagekraft dieser Metriken (Accuracy, Precision, Recall)?
# → Accuracy ermittelt den prozentualen Anteil der korrekten Vorhersagen (𝐓𝐏 + 𝐓𝐍) / (𝐓𝐏 + 𝐅𝐍 + 𝐅𝐏 + 𝐓𝐍).
# → Precision ist der Prozentsatz korrekter Antworten unter allen positiven Antworten TP/(TP+FP).
# → Recall ist der Prozentsatz korrekter Antworten unter allen positiven Eingaben. TP/(TP+TN)