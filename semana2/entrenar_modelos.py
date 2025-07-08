import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#crear carpeta para modelos
entrenamiento_dir = os.path.join("entrenamiento")
os.makedirs(entrenamiento_dir, exist_ok=True)

#leer los datos
df = pd.read_csv("../data_sintetica/clientes_compras_train.csv")

#variables
X = df.drop(columns=["DNI", "compra_estacional"])
y = df["compra_estacional"]

#columnas
categorical = ['sexo', 'temporada']
numeric = ['edad', 'frecuencia_ultimos_3m', 'gasto_promedio']

#preprocesamiento
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(), categorical)
])

#pipelines
pipe_svm = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True))
])

pipe_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

#Entrenamiento
pipe_svm.fit(X, y)
pipe_rf.fit(X, y)

#Guardar modelos
joblib.dump(pipe_svm, os.path.join(entrenamiento_dir, "modelo_svm.pkl"))
joblib.dump(pipe_rf, os.path.join(entrenamiento_dir, "modelo_rf.pkl"))

print("Modelos entrenados y guardados")