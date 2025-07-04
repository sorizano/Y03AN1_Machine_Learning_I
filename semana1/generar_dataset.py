# importando librerias
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

#Definir la carpeta de salida
output_dir = os.path.join(os.path.dirname(__file__), '..','data_sintetica') 
os.makedirs(output_dir, exist_ok=True)

#Generar datos sintéticos
X, y = make_classification(
    n_samples=500000,   # 500 mil registros
    n_features=10,      # 10 variables predictoras
    n_informative=6,    # 6 relevantes para la predicción
    n_redundant=2,       # 2 redundantes
    n_classes=2,         # Moroso o no
    weights=[0.7,0.3],   #70% no morosos, 30%morosos
    random_state=42
)

#Crear DataFrame
columnas = [f'feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columnas)
df['moroso'] = y

#Guardar en CSV
output_path = os.path.join(output_dir, 'dataset_morosidad.csv')
df.to_csv(output_path, index=False)

print(f"Dataset sentético generado con éxito en; \n{output_path}")