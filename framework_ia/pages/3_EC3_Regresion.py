"""
pages/3_EC3_Regresion.py
Caso de Estudio 3 — Regresión Supervisada: Lineal, Lasso, Ridge, SVR, DT, RF, GB
Requiere: modules/M_Caso.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from helpers import run_and_capture, display_results, section_header, \
                   warning_missing_module, warning_missing_data, metric_card

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="EC3 — Regresión", page_icon="📈", layout="wide")

st.markdown("""
<style>
.page-header {
    background: linear-gradient(90deg, #78350F, #D97706);
    color: white; padding: 1.2rem 1.8rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.page-header h2 { margin: 0; font-size: 1.5rem; font-weight: 700; }
.page-header p  { margin: .2rem 0 0 0; opacity: .8; font-size: .88rem; }
.metric-row { display:flex; gap:.6rem; margin-bottom:.8rem; flex-wrap:wrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>📈 EC3 — Regresión Supervisada</h2>
  <p>Lineal · Lasso · Ridge · SVR · Árbol DT · Random Forest · Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

# ── Module import ─────────────────────────────────────────────────────────────
MODULE_OK = False
Regresion = None

try:
    from M_Caso import Regresion
    MODULE_OK = True
except ImportError:
    try:
        from Ec3_main import Regresion
        MODULE_OK = True
    except ImportError:
        pass

if not MODULE_OK:
    warning_missing_module("M_Caso.py", "Caso de Estudio 3")
    st.markdown("""
    #### ¿Qué necesito copiar?
    Desde la carpeta **Caso de Estudio 3** copia estos archivos a `modules/`:
    | Archivo | Descripción |
    |---------|-------------|
    | `M_Caso.py` | Clase principal con todos los métodos de regresión |

    Reinicia la app después de copiar los archivos.
    """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración EC3")
    st.divider()

    dataset_opt = st.radio("Dataset", ["Toyota Price", "KC House Data"])

    TARGET_MAP = {
        "Toyota Price":  "precio",
        "KC House Data": "price",
    }
    DATASET_INFO = {
        "Toyota Price":  {"rows": 6_632,  "unit": "£"},
        "KC House Data": {"rows": 21_611, "unit": "$"},
    }

    st.divider()
    st.markdown("**Algoritmos**")
    run_lineal = st.checkbox("Regresión Lineal (Simple + Múltiple)", value=True)
    run_lasso  = st.checkbox("Lasso + LassoCV",    value=True)
    run_ridge  = st.checkbox("Ridge + RidgeCV",    value=True)
    run_svr    = st.checkbox("SVR",                value=False)
    run_dt_r   = st.checkbox("Árbol DT Regressor", value=True)
    run_rf_r   = st.checkbox("Random Forest Reg.", value=True)
    run_gb     = st.checkbox("Gradient Boosting",  value=False)

    st.divider()
    st.info("ℹ️ Los hiperparámetros están definidos internamente en M_Caso.py")

    st.divider()
    execute_btn = st.button("▶ Ejecutar regresión", type="primary",
                            use_container_width=True, disabled=not MODULE_OK)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILES = {
    "Toyota Price":  os.path.join(DATA_DIR, "Toyota_Price.csv"),
    "KC House Data": os.path.join(DATA_DIR, "kc_house_data.csv"),
}

# ── Session state ─────────────────────────────────────────────────────────────
for k in ["ec3_df", "ec3_results"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Load helper ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset_ec3(path):
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, decimal=".")
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    df.dropna(inplace=True)
    return df


# ── Execute ───────────────────────────────────────────────────────────────────
if execute_btn and MODULE_OK:
    path = DATA_FILES[dataset_opt]
    if not os.path.exists(path):
        warning_missing_data(os.path.basename(path), "Caso de Estudio 3")
    else:
        with st.spinner("Cargando y limpiando datos..."):
            df_raw = load_dataset_ec3(path)
            st.session_state.ec3_df = df_raw

        # ── Instanciar Regresion y aplicar limpieza específica del dataset ────
        reg = Regresion(df_raw.copy())

        try:
            if dataset_opt == "Toyota Price":
                df_clean = reg.dataCleaning()   # codifica modelo/transmision/tipo_combustible
            else:
                df_clean = reg.dataCleaning2()  # codifica floors/waterfront/view/condition/grade/zipcode
        except Exception as e:
            st.warning(f"dataCleaning falló ({e}). Usando solo columnas numéricas.")
            df_clean = df_raw.select_dtypes(include="number").dropna()

        # ── Preparar X / y y dividir en train-test ───────────────────────────
        target_col = TARGET_MAP[dataset_opt]
        if target_col not in df_clean.columns:
            st.error(f"Columna objetivo '{target_col}' no encontrada tras la limpieza.")
            st.stop()

        X_all = df_clean.drop(columns=[target_col])
        y     = df_clean[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.25, random_state=42
        )

        # RegLinSimp requiere una sola variable: la de mayor correlación con y
        best_feat  = X_all.corrwith(y).abs().idxmax()
        X_train_1d = X_train[best_feat].values          # 1-D → reshape(-1,1) interno en M_Caso
        X_test_2d  = X_test[best_feat].values.reshape(-1, 1)  # 2-D para predict

        results = {}

        def _exec(name, func, *args):
            with st.spinner(f"Ejecutando {name}..."):
                figs, txt = run_and_capture(func, *args)
                results[name] = {"figs": figs, "text": txt}

        if run_lineal:
            _exec(f"Regresión Lineal Simple  ({best_feat})",
                  reg.RegLinSimp,
                  X_train_1d, X_test_2d, y_train.values, y_test.values)
            _exec("Regresión Lineal Múltiple",
                  reg.RegLinMult,
                  X_train, X_test, y_train, y_test)

        if run_lasso:
            _exec("Lasso",   reg.RegLasso,   X_train, X_test, y_train, y_test)
            _exec("LassoCV", reg.RegLassoCV, X_train, X_test, y_train, y_test)

        if run_ridge:
            _exec("Ridge",   reg.RegRidge,   X_train, X_test, y_train, y_test)
            _exec("RidgeCV", reg.RegRidgeCV, X_train, X_test, y_train, y_test)

        if run_svr:
            _exec("SVR  (rbf + linear + poly)",
                  reg.SVM, X_train, X_test, y_train, y_test)

        if run_dt_r:
            _exec("Árbol DT Regressor",
                  reg.DecisionTreeReg, X_train, X_test, y_train, y_test)

        if run_rf_r:
            _exec("Random Forest Regressor",
                  reg.RandomForestReg, X_train, X_test, y_train, y_test)

        if run_gb:
            _exec("Gradient Boosting Regressor",
                  reg.XGBoostingReg, X_train, X_test, y_train, y_test)

        st.session_state.ec3_results = results
        st.success(f"✅ {len(results)} modelo(s) ejecutados.")


# ── Display ───────────────────────────────────────────────────────────────────
if st.session_state.ec3_df is not None:
    df   = st.session_state.ec3_df
    info = DATASET_INFO[dataset_opt]

    section_header("📋", "Vista del Dataset",
                   f"{dataset_opt} — {df.shape[0]:,} filas · {df.shape[1]} columnas")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(df.head(8), use_container_width=True)
    with col_b:
        target_col = TARGET_MAP[dataset_opt]
        if target_col in df.columns:
            fig_t, ax_t = plt.subplots(figsize=(4, 3))
            df[target_col].hist(bins=40, ax=ax_t, color="#D97706", edgecolor="white")
            ax_t.set_title(f"Distribución de {target_col}")
            ax_t.set_xlabel(f"Precio ({info['unit']})")
            plt.tight_layout()
            st.pyplot(fig_t, use_container_width=True)
            plt.close(fig_t)

            st.markdown(
                metric_card("Media",  f"{info['unit']}{df[target_col].mean():,.0f}", color="#D97706"),
                unsafe_allow_html=True
            )
            st.markdown(
                metric_card("Máximo", f"{info['unit']}{df[target_col].max():,.0f}", color="#DC2626"),
                unsafe_allow_html=True
            )

    st.divider()

    if st.session_state.ec3_results:
        for model_name, data in st.session_state.ec3_results.items():
            st.markdown(f"#### 📐 {model_name}")
            display_results(data["figs"], data["text"],
                            cols=2, section_title=f"{model_name} — gráficos")
            st.divider()

elif not MODULE_OK:
    pass
else:
    st.info("👈 Configura los parámetros en el sidebar y presiona **▶ Ejecutar regresión**.")

with st.expander("💡 Métricas de evaluación utilizadas"):
    st.markdown("""
    | Métrica | Descripción |
    |---------|-------------|
    | **RMSE** | Raíz del Error Cuadrático Medio — penaliza errores grandes |
    | **MAE**  | Error Absoluto Medio — robusto a outliers |
    | **ER**   | Error Relativo — útil para comparar entre datasets |
    | **R²**   | Varianza explicada. R²=1 es ajuste perfecto; R²≤0 peor que la media |
    """)