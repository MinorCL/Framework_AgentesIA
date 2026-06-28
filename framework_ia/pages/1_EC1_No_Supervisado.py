"""
pages/1_EC1_No_Supervisado.py
Caso de Estudio 1 — Métodos No Supervisados: ACP, HAC, KMeans
"""
# DESPUÉS
import sys, os

# Path absoluto basado en la ubicación real del proyecto
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, 'modules'))
sys.path.insert(0, os.path.join(_BASE, 'utils'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from helpers import run_and_capture, display_results, section_header, warning_missing_module, warning_missing_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="EC1 — No Supervisado", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.page-header {
    background: linear-gradient(90deg, #4C1D95, #7C3AED);
    color: white; padding: 1.2rem 1.8rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.page-header h2 { margin: 0; font-size: 1.5rem; font-weight: 700; }
.page-header p  { margin: .2rem 0 0 0; opacity: .8; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>🔍 EC1 — Métodos No Supervisados</h2>
  <p>ACP · Agrupamiento Jerárquico (HAC) · KMeans · KMedoids</p>
</div>
""", unsafe_allow_html=True)

# ── Module import ─────────────────────────────────────────────────────────────
try:
    from PaqNOSup import analisisEDA, NoSupervisado, ACPmain
    MODULE_OK = True
except Exception as e:
    MODULE_OK = False
    st.error(f"Error al importar PaqNOSup: {type(e).__name__}: {e}")
    warning_missing_module("PaqNOSup.py", "Caso de Estudio 1 / Caso de Estudio 4")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración EC1")
    st.divider()

    dataset_opt = st.radio(
        "Dataset",
        ["Hotel Bookings", "Bank Churners"],
        help="Selecciona el conjunto de datos a analizar"
    )

    st.divider()
    st.markdown("**Preprocesamiento**")
    drop_nulls   = st.checkbox("Eliminar nulos", value=True)
    only_numeric = st.checkbox("Solo variables numéricas", value=True)
    use_dummies  = st.checkbox("Convertir categóricas (dummies)", value=False,
                               disabled=only_numeric,
                               help="Se desactiva si 'Solo numéricas' está activo")
    st.divider()
    st.markdown("**Análisis**")
    run_eda     = st.checkbox("EDA (boxplots, histogramas)", value=False)
    run_corr    = st.checkbox("Mapa de correlación", value=False)
    run_acp     = st.checkbox("ACP", value=True)
    n_comp      = st.slider("Componentes ACP", 2, 10, 2, disabled=not run_acp)
    run_hac     = st.checkbox("Agrupamiento Jerárquico (HAC)", value=False,
                              help="⚠️ Puede ser lento en datasets grandes")
    run_kmeans  = st.checkbox("KMeans / KMedoids", value=True)

    st.divider()
    execute_btn = st.button("▶ Ejecutar análisis", type="primary", use_container_width=True)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILES = {
    "Hotel Bookings": (os.path.join(DATA_DIR, "hotel_bookings_muestra.csv"), 1),
    "Bank Churners":  (os.path.join(DATA_DIR, "BankChurners.csv"),           1),
}

# ── Session state keys ────────────────────────────────────────────────────────
KEYS = ["ec1_df", "ec1_df_scaled", "ec1_eda_figs", "ec1_eda_text",
        "ec1_corr_figs", "ec1_acp_figs", "ec1_acp_text",
        "ec1_hac_figs", "ec1_hac_text", "ec1_km_figs", "ec1_km_text"]
for k in KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Load & preprocess helper ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(path, sep_num, drop_n, only_num, dummies):
    df_raw = pd.read_csv(path, sep="," if sep_num == 1 else ";",
                         decimal=".", index_col=0 if sep_num == 1 else None)
    df = df_raw.copy()

    if drop_n:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

    if only_num:
        df = df.select_dtypes(include=["number"])
    elif dummies:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True).astype(float)

    df.dropna(inplace=True)  # Seguridad final

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df, df_scaled


# ── Execute ───────────────────────────────────────────────────────────────────
if execute_btn:
    if not MODULE_OK:
        st.error("No se puede ejecutar: módulo PaqNOSup no disponible.")
    else:
        path, sep_num = DATA_FILES[dataset_opt]
        if not os.path.exists(path):
            warning_missing_data(os.path.basename(path), f"Caso de Estudio 1")
        else:
            with st.spinner(f"Cargando {dataset_opt}..."):
                try:
                    df, df_scaled = load_and_preprocess(
                        path, sep_num, drop_nulls, only_numeric, use_dummies
                    )
                    st.session_state.ec1_df = df
                    st.session_state.ec1_df_scaled = df_scaled
                except Exception as e:
                    st.error(f"Error al cargar datos: {e}")
                    st.stop()

            nos = NoSupervisado(df_scaled)

            # EDA
            if run_eda:
                with st.spinner("Generando EDA..."):
                    def _eda():
                        eda_obj = analisisEDA.__new__(analisisEDA)
                        eda_obj._analisisEDA__df = df  # acceso al df privado
                        eda_obj.graficoBoxplot()
                        eda_obj.histogramas()

                    figs, txt = run_and_capture(_eda)
                    st.session_state.ec1_eda_figs = figs
                    st.session_state.ec1_eda_text = txt

            # Correlación
            if run_corr:
                with st.spinner("Calculando correlaciones..."):
                    def _corr():
                        eda_obj = analisisEDA.__new__(analisisEDA)
                        eda_obj._analisisEDA__df = df
                        eda_obj.graficoCorrelacion()

                    figs, txt = run_and_capture(_corr)
                    st.session_state.ec1_corr_figs = figs
                    st.session_state.ec1_corr_text = txt

            # ACP
            if run_acp:
                with st.spinner("Ejecutando ACP..."):
                    figs, txt = run_and_capture(nos.ACP, n_comp)
                    st.session_state.ec1_acp_figs = figs
                    st.session_state.ec1_acp_text = txt

            # HAC
            if run_hac:
                if len(df_scaled) > 2000:
                    st.warning(f"El dataset tiene {len(df_scaled)} filas. HAC puede ser muy lento. "
                               "Considera usar una muestra.")
                with st.spinner("Ejecutando HAC (puede tardar)..."):
                    sample = df_scaled.sample(min(500, len(df_scaled)), random_state=42)
                    nos_sample = NoSupervisado(sample)
                    figs, txt = run_and_capture(nos_sample.HAC)
                    st.session_state.ec1_hac_figs = figs
                    st.session_state.ec1_hac_text = txt

            # KMeans
            if run_kmeans:
                with st.spinner("Ejecutando KMeans / KMedoids..."):
                    figs, txt = run_and_capture(nos.KMedia)
                    st.session_state.ec1_km_figs = figs
                    st.session_state.ec1_km_text = txt

            st.success("✅ Análisis completado.")

# ── Display ───────────────────────────────────────────────────────────────────
if st.session_state.ec1_df is not None:
    df        = st.session_state.ec1_df
    df_scaled = st.session_state.ec1_df_scaled

    # Dataset overview
    section_header("📋", "Vista del Dataset", f"{dataset_opt} — {df.shape[0]} filas · {df.shape[1]} columnas")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(df.head(10), use_container_width=True)
    with col_b:
        st.markdown("**Estadísticas descriptivas**")
        st.dataframe(df.describe().round(3), use_container_width=True)

    st.divider()

    # EDA
    if st.session_state.ec1_eda_figs is not None:
        section_header("📊", "Análisis Exploratorio (EDA)")
        display_results(st.session_state.ec1_eda_figs, st.session_state.ec1_eda_text or "", cols=3)

    # Correlación
    if st.session_state.get("ec1_corr_figs") is not None:
        section_header("🔥", "Mapa de Correlación")
        display_results(st.session_state.ec1_corr_figs, "", cols=1)

    # ACP
    if st.session_state.ec1_acp_figs is not None:
        section_header("🔵", "Análisis de Componentes Principales (ACP)",
                       f"n_componentes = {n_comp}")
        display_results(st.session_state.ec1_acp_figs, st.session_state.ec1_acp_text or "", cols=3)

    # HAC
    if st.session_state.ec1_hac_figs is not None:
        section_header("🌿", "Agrupamiento Jerárquico (HAC)",
                       "Muestra de 500 filas para optimizar rendimiento")
        display_results(st.session_state.ec1_hac_figs, st.session_state.ec1_hac_text or "", cols=2)

    # KMeans
    if st.session_state.ec1_km_figs is not None:
        section_header("⭕", "KMeans / KMedoids")
        display_results(st.session_state.ec1_km_figs, st.session_state.ec1_km_text or "", cols=2)

else:
    st.info("👈 Configura los parámetros en el sidebar y presiona **▶ Ejecutar análisis**.")