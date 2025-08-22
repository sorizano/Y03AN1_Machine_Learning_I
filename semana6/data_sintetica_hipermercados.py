import os, random
import numpy as np
import pandas as pd
from pathlib import Path
# -------------------
# Configuración
# -------------------
RANDOM_SEED = 42
N_FILAS = 50_000
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------
# Ubicaciones
# -------------------
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()   # si se ejecuta desde notebook

ROOT = THIS_DIR.parent     # carpeta raíz del proyecto
DATA_ROOT = ROOT / "data_sintetica"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
PKL_PATH = DATA_ROOT / "hipermercados.pkl"

# -------------------
# Catálogos
# -------------------
TIENDAS = ["Hiper Urbano", "Hiper Suburbano", "Hiper Vecinal"]
SEGMENTO_INGRESO = ["Bajo", "Medio", "Alto"]
CLIMA = ["Soleado", "Nublado", "Lluvioso"]
PROMO_TIPO = ["2x1", "Descuento", "Combo", "Ninguna"]

def sample_categorical(opts, p=None):
    if p is None:
        return random.choice(opts)
    return np.random.choice(opts, p=p)

# -------------------
# Generador de filas
# -------------------
def generar_fila(i: int) -> dict:
    seg = sample_categorical(SEGMENTO_INGRESO, p=[0.35, 0.45, 0.20])
    puntos_lealtad = np.clip(int(np.random.gamma(2.0, 40)), 0, 1000)
    frecuencia_hist = np.clip(int(np.random.normal(6, 2)), 0, 30)
    tienda = sample_categorical(TIENDAS)
    clima = sample_categorical(CLIMA)
    es_fin_semana = int(np.random.rand() < 0.35)
    hora = np.random.randint(8, 23)
    promo = sample_categorical(PROMO_TIPO)
    cabecera_gondola = int(np.random.rand() < 0.5)
    cross_selling = int(np.random.rand() < 0.55)
    degustacion = int(np.random.rand() < 0.15)
    trafico = int(np.clip(np.random.normal(250, 60), 50, 600))

    base_ticket = {"Bajo": 45, "Medio": 70, "Alto": 120}[seg]
    base_items = {"Bajo": 5, "Medio": 8, "Alto": 12}[seg]

    uplift_items, uplift_ticket = 0, 0
    if promo == "2x1":
        uplift_items += 1.5
        uplift_ticket += 12
    if cabecera_gondola:
        uplift_items += 1
    if cross_selling:
        uplift_ticket += 7

    n_items = int(np.clip(np.random.normal(base_items + uplift_items, 1.8), 1, 60))
    ticket = float(np.clip(np.random.normal(base_ticket + uplift_ticket, 12.0), 5, 1200))

    compra_impulsiva = int(ticket > base_ticket * 1.2)
    ticket_alto = int(ticket > base_ticket * 1.5)

    return {
        "id_tx": i + 1,
        "tienda": tienda,
        "segmento_ingreso": seg,
        "puntos_lealtad": puntos_lealtad,
        "frecuencia_hist": frecuencia_hist,
        "clima": clima,
        "fin_semana": es_fin_semana,
        "hora": hora,
        "trafico_tienda": trafico,
        "promo_tipo": promo,
        "cabecera_gondola": cabecera_gondola,
        "cross_selling": cross_selling,
        "degustacion": degustacion,
        "n_items": n_items,
        "ticket_soles": round(ticket, 2),
        "compra_impulsiva": compra_impulsiva,
        "ticket_alto": ticket_alto,
    }

# -------------------
# Main
# -------------------
def main():
    filas = [generar_fila(i) for i in range(N_FILAS)]
    df = pd.DataFrame(filas)
    df.to_pickle(PKL_PATH)
    print(f"✅ PKL guardado en: {PKL_PATH} | Filas: {len(df):,}")

if __name__ == "__main__":
    main()