import pandas as pd
import numpy as np
import os

# Crear carpeta si no existe
os.makedirs('../data_sintetica', exist_ok=True)

# Parámetros para generación aleatoria
np.random.seed(42)
n = 300000  # Número de muestras para entrenamiento

# Opciones de mineral y transporte
minerales = ['cobre', 'plomo', 'zinc']
transportes = ['tren', 'camion']

# Generar data sintética para entrenamiento
data = pd.DataFrame({
    'mineral': np.random.choice(minerales, n),
    'produccion_diaria_tn': np.random.randint(500, 5000, size=n),  # entre 500 y 5000 toneladas por día
    'tiempo_a_fundicion': np.random.uniform(2, 12, size=n).round(2),  # en horas
    'tiempo_a_puerto': np.random.uniform(5, 20, size=n).round(2),     # en horas
    'distancia_total_km': np.random.randint(100, 1000, size=n)        # distancia total estimada
})

# Lógica sintética para decidir transporte
def decidir_transporte(row):
    if row['distancia_total_km'] > 600 or row['produccion_diaria_tn'] > 3000:
        return 'tren'
    else:
        return 'camion'

data['transporte'] = data.apply(decidir_transporte, axis=1)

# Guardar datos de entrenamiento
data.to_csv('../data_sintetica/data_sintetica_sem3.csv', index=False)
print("Archivo 'data_sintetica.csv_sem3' generado.")

# Generar nuevos datos SIN transporte (para predecir)
nuevos_datos = pd.DataFrame({
    'mineral': np.random.choice(minerales, 20),
    'produccion_diaria_tn': np.random.randint(500, 5000, size=20),
    'tiempo_a_fundicion': np.random.uniform(2, 12, size=20).round(2),
    'tiempo_a_puerto': np.random.uniform(5, 20, size=20).round(2),
    'distancia_total_km': np.random.randint(100, 1000, size=20)
})

nuevos_datos.to_csv('../data_sintetica/nuevos_datos_sem3.csv', index=False)
print("Archivo 'nuevos_datos_sem3.csv' generado.")
