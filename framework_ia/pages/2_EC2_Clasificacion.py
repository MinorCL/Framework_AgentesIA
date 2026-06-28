"""
pages/2_EC2_Clasificacion.py
Caso de Estudio 2 — Clasificación Supervisada: KNN, DT, RF, XGBoost, ADABoost
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

from helpers import run_and_capture, display_results, section_header, \
                   warning_missing_module, warning_missing_data, metric_card

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="EC2 — Clasificación", page_icon="🌳", layout="wide")

st.markdown("""
<style>
.page-header {
    background: linear-gradient(90deg, #064E3B, #059669);
    color: white; padding: 1.2rem 1.8rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.page-header h2 { margin: 0; font-size: 1.5rem; font-weight: 700; }
.page-header p  { margin: .2rem 0 0 0; opacity: .8; font-size: .88rem; }
.result-box {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 1rem; margin-bottom: .8rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>🌳 EC2 — Clasificación Supervisada</h2>
  <p>KNN · Árbol de Decisión · Random Forest · XGBoost · ADABoost</p>
</div>
""", unsafe_allow_html=True)

# ── Module import ─────────────────────────────────────────────────────────────
try:
    from GuiaClaseSupervisada import Supervisado
    MODULE_OK = True
except ImportError:
    MODULE_OK = False
    warning_missing_module("GuiaClaseSupervisada.py", "Caso de Estudio 2 / Caso de Estudio 4")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración EC2")
    st.divider()

    dataset_opt = st.radio("Dataset", ["Potabilidad", "Diabetes"])

    TARGET_MAP = {
        "Potabilidad": "Potability",
        "Diabetes":    "class",       # ajusta según tu CSV
    }

    st.divider()
    st.markdown("**Algoritmos a ejecutar**")
    run_knn  = st.checkbox("KNN",         value=True)
    run_dt   = st.checkbox("Árbol DT",    value=True)
    run_rf   = st.checkbox("Random Forest", value=True)
    run_xg   = st.checkbox("XGBoost",     value=False)
    run_ada  = st.checkbox("ADABoost",    value=False)
    run_bm   = st.checkbox("Benchmark comparativo", value=False,
                           help="Ejecuta todos los modelos con parámetros por defecto")

    st.divider()
    st.markdown("**Hiperparámetros**")
    with st.expander("KNN"):
        knn_k = st.slider("k (vecinos)", 1, 25, 5, key="knn_k")
    with st.expander("Árbol DT"):
        dt_depth = st.slider("max_depth", 1, 20, 5, key="dt_depth")
        dt_split = st.slider("min_samples_split", 2, 50, 2, key="dt_split")
    with st.expander("Random Forest"):
        rf_n   = st.slider("n_estimators", 10, 300, 100, step=10, key="rf_n")
        rf_d   = st.slider("max_depth", 1, 20, 4, key="rf_d")
        rf_s   = st.slider("min_samples_split", 2, 50, 2, key="rf_s")
    with st.expander("XGBoost / ADABoost"):
        xg_n   = st.slider("n_estimators", 10, 300, 100, step=10, key="xg_n")
        xg_d   = st.slider("max_depth", 1, 10, 4, key="xg_d")
        xg_s   = st.slider("min_samples_split", 2, 50, 2, key="xg_s")
        ada_n  = st.slider("ADA n_estimators", 10, 300, 100, step=10, key="ada_n")

    st.divider()
    execute_btn = st.button("▶ Ejecutar clasificación", type="primary", use_container_width=True)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILES = {
    "Potabilidad": os.path.join(DATA_DIR, "potabilidad_V2.csv"),
    "Diabetes":    os.path.join(DATA_DIR, "diabetes_V2.csv"),
}

# ── Session state ─────────────────────────────────────────────────────────────
for k in ["ec2_df", "ec2_results"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Load helper ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(path, target_col):
    """Carga CSV y renombra la columna objetivo a 'target'."""
    # Intenta separadores comunes
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, decimal=".")
            if len(df.columns) > 1:
                break
        except Exception:
            continue

    # Renombrar target si existe
    if target_col in df.columns:
        df.rename(columns={target_col: "target"}, inplace=True)
    elif "target" not in df.columns:
        # Asume que la última columna es el target
        df.rename(columns={df.columns[-1]: "target"}, inplace=True)


    df.dropna(inplace=True)

    if not pd.api.types.is_numeric_dtype(df["target"]):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["target"])
    else:
        df["target"] = df["target"].astype(int)
    return df


# ── Execute ───────────────────────────────────────────────────────────────────
if execute_btn:
    if not MODULE_OK:
        st.error("Módulo GuiaClaseSupervisada no disponible.")
    else:
        path = DATA_FILES[dataset_opt]
        if not os.path.exists(path):
            warning_missing_data(os.path.basename(path), "Caso de Estudio 2")
        else:
            with st.spinner("Cargando datos..."):
                df = load_dataset(path, TARGET_MAP[dataset_opt])
                st.session_state.ec2_df = df

            sup = Supervisado(df)
            results = {}

            if run_knn:
                with st.spinner("Ejecutando KNN..."):
                    figs, txt = run_and_capture(sup.KNN, knn_k)
                    results["KNN"] = {"figs": figs, "text": txt,
                                      "params": f"k={knn_k}"}

            if run_dt:
                with st.spinner("Ejecutando Árbol DT..."):
                    figs, txt = run_and_capture(sup.DT, dt_split, dt_depth)
                    results["Árbol DT"] = {"figs": figs, "text": txt,
                                           "params": f"max_depth={dt_depth}, min_split={dt_split}"}

            if run_rf:
                with st.spinner("Ejecutando Random Forest..."):
                    figs, txt = run_and_capture(sup.RF, rf_n, rf_s, rf_d)
                    results["Random Forest"] = {"figs": figs, "text": txt,
                                                "params": f"n={rf_n}, max_depth={rf_d}"}

            if run_xg:
                with st.spinner("Ejecutando XGBoost..."):
                    figs, txt = run_and_capture(sup.XG, xg_n, xg_s, xg_d)
                    results["XGBoost"] = {"figs": figs, "text": txt,
                                          "params": f"n={xg_n}, max_depth={xg_d}"}

            if run_ada:
                with st.spinner("Ejecutando ADABoost..."):
                    figs, txt = run_and_capture(sup.ADA, ada_n)
                    results["ADABoost"] = {"figs": figs, "text": txt,
                                           "params": f"n={ada_n}"}

            if run_bm:
                with st.spinner("Ejecutando Benchmark..."):
                    figs, txt = run_and_capture(sup.BM)
                    results["Benchmark"] = {"figs": figs, "text": txt, "params": "default"}

            st.session_state.ec2_results = results
            st.success(f"✅ {len(results)} algoritmo(s) ejecutados.")

# ── Display ───────────────────────────────────────────────────────────────────
if st.session_state.ec2_df is not None:
    df = st.session_state.ec2_df

    # Dataset overview
    section_header("📋", "Vista del Dataset",
                   f"{dataset_opt} — {df.shape[0]} filas · {df.shape[1]} cols")

    target_counts = df["target"].value_counts()
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.dataframe(df.head(8), use_container_width=True)
    with col_b:
        st.markdown("**Distribución de clases**")
        fig_dist, ax = plt.subplots(figsize=(4, 3))
        target_counts.plot(kind="bar", ax=ax, color=["#059669", "#EF4444"])
        ax.set_xlabel("Clase")
        ax.set_ylabel("Conteo")
        ax.set_title("target")
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)
    with col_c:
        st.markdown("**Clases**")
        for cls, cnt in target_counts.items():
            pct = 100 * cnt / len(df)
            st.markdown(
                metric_card(f"Clase {cls}", f"{cnt:,}", f"{pct:.1f}%",
                            "#059669" if cls == 0 else "#EF4444"),
                unsafe_allow_html=True
            )

    st.divider()

    # Results per algorithm
    if st.session_state.ec2_results:
        for algo_name, data in st.session_state.ec2_results.items():
            st.markdown(f"#### 🔬 {algo_name}  `{data['params']}`")
            display_results(data["figs"], data["text"],
                            cols=2, section_title=f"{algo_name} — gráficos")
            st.divider()
else:
    st.info("👈 Configura los parámetros en el sidebar y presiona **▶ Ejecutar clasificación**.")

# ── Tip ───────────────────────────────────────────────────────────────────────
with st.expander("💡 Notas sobre los resultados"):
    st.markdown("""
    - **PG** = Precisión Global  
    - **EG** = Error Global  
    - **PP / PN** = Precisión por categoría positiva / negativa  
    - Los resultados varían entre ejecuciones por la partición aleatoria train/test.  
    - Usa **Benchmark** para comparar todos los modelos con parámetros por defecto.
    """)