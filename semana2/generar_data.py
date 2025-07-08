import pandas as pd
import numpy as np
import os

# Crear directorio si no existe
os.makedirs("../data_sintetica", exist_ok=True)

# Semilla para reproducibilidad
np.random.seed(42)

# Generar data sintética de entrenamiento
def generar_data(n=500):
    data = pd.DataFrame({
        'DNI': np.random.randint(10000000, 99999999, n),
        'sexo': np.random.choice(['M', 'F'], size=n),
        'edad': np.random.randint(18, 65, n),
        'frecuencia_ultimos_3m': np.random.randint(0, 10, n),
        'gasto_promedio': np.round(np.random.uniform(20, 300, n), 2),
        'temporada': np.random.choice(['navidad', 'clases', 'fiestas'], size=n),
    })

    # Regla simple para generar la etiqueta
    data['compra_estacional'] = (
        ((data['temporada'] == 'navidad') & (data['gasto_promedio'] > 150)) |
        ((data['temporada'] == 'clases') & (data['edad'] < 35)) |
        ((data['frecuencia_ultimos_3m'] > 5))
    ).astype(int)

    return data

# Crear train y test
train = generar_data(500)
test = generar_data(100)

# Guardar
train.to_csv("../data_sintetica/clientes_compras_train.csv", index=False)
test.to_csv("../data_sintetica/clientes_compras_test.csv", index=False)

print("✅ Datos sintéticos generados.")
