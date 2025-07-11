#importar las librerias
import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

#Ruta base del proyecto
ruta_base = os.path.dirname(os.path.dirname(__file__))
ruta_csv = os.path.join(ruta_base, 'data_sintetica', 'farmacologia_recuperacion.csv')

#Cargar datos
df = pd.read_csv(ruta_csv)
X = df[['edad', 'peso', 'dosis']]
y = df['tiempo_recuperacion']

#transformación polinómica de grado 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

#Separación de train/test
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#Entrenar el modelo
modelo_poly = LinearRegression()
modelo_poly.fit(X_train, y_train)

#Guardar el modelo
ruta_modelos = os.path.join(os.path.dirname(__file__), 'entrenamiento')
os.makedirs(ruta_modelos, exist_ok=True)

joblib.dump(modelo_poly, os.path.join(ruta_modelos, 'regresion_polinimica.pkl'))
joblib.dump(poly, os.path.join(ruta_modelos, 'poly_transform.pkl'))

#Evaluar modelo
y_pred = modelo_poly.predict(X_test)
print(f"R2: {r2_score(y_test, y_pred):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")