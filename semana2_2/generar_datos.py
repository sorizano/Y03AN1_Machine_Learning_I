import pandas as pd
import numpy as np
import os

# ✅ Crear carpeta de salida si no existe
output_path = os.path.join(os.path.dirname(__file__), '..', 'data_sintetica')
os.makedirs(output_path, exist_ok=True)

# 🎲 Semilla para reproducibilidad
np.random.seed(42)
n = 200  # muestras

# 🧪 Variables
edad = np.random.randint(18, 90, n)
peso = np.random.randint(50, 120, n)
dosis_base = np.random.uniform(0.5, 2.0, n)
ruido = np.random.normal(0, 2, n)

# 🎯 Target: tiempo de recuperación
tiempo_recuperacion = 0.1 * edad + 0.05 * peso - 5 * dosis_base + ruido

# 📊 DataFrame final
df_sintetico = pd.DataFrame({
    'edad': edad,
    'peso': peso,
    'dosis': dosis_base,
    'tiempo_recuperacion': tiempo_recuperacion
})

# 💾 Guardar archivo CSV
file_path = os.path.join(output_path, "farmacologia_recuperacion.csv")
df_sintetico.to_csv(file_path, index=False)
print(f"✅ Datos guardados en: {file_path}")