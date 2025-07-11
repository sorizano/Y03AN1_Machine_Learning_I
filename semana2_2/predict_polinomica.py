import pandas as pd
import joblib
import os

#Ruta Base
ruta_base = os.path.dirname(os.path.dirname(__file__))

#Cargar el modelo y transformador
ruta_modelo = os.path.join(ruta_base, 'semana2_2', 'entrenamiento', 'regresion_polinimica.pkl')
ruta_poly = os.path.join(ruta_base, 'semana2_2', 'entrenamiento', 'poly_transform.pkl')
modelo = joblib.load(ruta_modelo)
poly = joblib.load(ruta_poly)

#Cargar nuevos datos
ruta_csv = os.path.join(ruta_base, 'data_sintetica', 'farmacologia_recuperacion.csv')
df = pd.read_csv(ruta_csv)

#Seleccionar variable
X_new = df[['edad', 'peso', 'dosis']]
X_poly = poly.transform(X_new)

# Hacer predicción
df['pred_polinomica'] = modelo.predict(X_poly)

# Guargar resultado
ruta_resultado = os.path.join(ruta_base, 'resultado de modelos')
os.makedirs(ruta_resultado, exist_ok=True)
df.to_excel(os.path.join(ruta_resultado, 'predicciones_polinomica.xlsx'), index=False)

print("Predicciones polinómicas guardadas correctamente")