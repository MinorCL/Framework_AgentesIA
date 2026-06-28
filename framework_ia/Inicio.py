import streamlit as st

st.set_page_config(
    page_title="EIF420O — Inteligencia Artificial",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #2563EB 100%);
    color: white; padding: 2.5rem 2.5rem; border-radius: 16px;
    text-align: center; margin-bottom: 2rem;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0 0 0.3rem 0; letter-spacing: -1px; }
.hero h2 { font-size: 1.3rem; font-weight: 300; margin: 0 0 1rem 0; opacity: 0.85; }
.hero p  { font-size: 0.9rem; opacity: 0.75; margin: 0; }

.card {
    border-radius: 12px; padding: 1.4rem 1.5rem; border: 1px solid #E2E8F0;
    height: 100%; transition: box-shadow .2s;
}
.card:hover { box-shadow: 0 6px 20px -4px rgba(0,0,0,.12); }
.c1 { border-top: 4px solid #7C3AED; background:#FAF5FF; }
.c2 { border-top: 4px solid #059669; background:#F0FDF4; }
.c3 { border-top: 4px solid #D97706; background:#FFFBEB; }
.c4 { border-top: 4px solid #DC2626; background:#FFF1F2; }

.card-num { font-size: 1rem; font-weight: 700; letter-spacing: 1px; margin-bottom: .4rem; }
.card-title { font-size: 1rem; font-weight: 600; margin-bottom: .6rem; color: #1E293B; }
.card-desc { font-size: .82rem; color: #475569; line-height: 1.5; }
.badge {
    display:inline-block; padding:.15rem .55rem; border-radius:999px;
    font-size:.72rem; font-weight:600; margin:.15rem .15rem .15rem 0;
}
.b-purple{background:#EDE9FE;color:#5B21B6}
.b-green {background:#D1FAE5;color:#065F46}
.b-amber {background:#FEF3C7;color:#92400E}
.b-red   {background:#FEE2E2;color:#991B1B}

.team-box {
    background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px;
    padding:1.2rem 1.5rem; margin-top:1.5rem;
}
.how-to {
    background:#EFF6FF; border-left:4px solid #3B82F6; border-radius:8px;
    padding:1rem 1.2rem; margin-top:1rem; font-size:.88rem; color:#1E40AF;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 EIF420O</h1>
  <h2>Inteligencia Artificial — Framework Interactivo</h2>
  <p>Universidad Nacional de Costa Rica &nbsp;·&nbsp; Escuela de Informática &nbsp;·&nbsp; I Semestre 2026</p>
  <p style="margin-top:.4rem"><strong>Profesor:</strong> Dr. Juan Murillo Morera</p>
</div>
""", unsafe_allow_html=True)

# ── Case Study Cards ──────────────────────────────────────────────────────────
st.markdown("### 📂 Casos de Estudio")
st.caption("Navega a cada caso de estudio usando el menú lateral izquierdo.")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card c1">
      <div class="card-num" style="color:#7C3AED">EC 1</div>
      <div class="card-title">Métodos No Supervisados</div>
      <div class="card-desc">
        Reducción de dimensionalidad y agrupamiento sin etiquetas.<br><br>
        <span class="badge b-purple">ACP</span>
        <span class="badge b-purple">HAC</span>
        <span class="badge b-purple">KMeans</span>
        <span class="badge b-purple">TSNE</span>
        <span class="badge b-purple">UMAP</span>
      </div>
      <div style="margin-top:.8rem;font-size:.8rem;color:#6B7280">
        🗃️ Hotel Bookings &nbsp;·&nbsp; Bank Churners
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card c2">
      <div class="card-num" style="color:#059669">EC 2</div>
      <div class="card-title">Clasificación Supervisada</div>
      <div class="card-desc">
        Predicción de categorías con modelos supervisados.<br><br>
        <span class="badge b-green">KNN</span>
        <span class="badge b-green">Árbol DT</span>
        <span class="badge b-green">Random Forest</span>
        <span class="badge b-green">XGBoost</span>
        <span class="badge b-green">ADABoost</span>
      </div>
      <div style="margin-top:.8rem;font-size:.8rem;color:#6B7280">
        🗃️ Potabilidad &nbsp;·&nbsp; Diabetes
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card c3">
      <div class="card-num" style="color:#D97706">EC 3</div>
      <div class="card-title">Regresión Supervisada</div>
      <div class="card-desc">
        Predicción de valores continuos con múltiples algoritmos.<br><br>
        <span class="badge b-amber">Lineal</span>
        <span class="badge b-amber">Lasso</span>
        <span class="badge b-amber">Ridge</span>
        <span class="badge b-amber">SVR</span>
        <span class="badge b-amber">RF</span>
        <span class="badge b-amber">Gradient Boosting</span>
      </div>
      <div style="margin-top:.8rem;font-size:.8rem;color:#6B7280">
        🗃️ Toyota Price &nbsp;·&nbsp; KC House Data
      </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card c4">
      <div class="card-num" style="color:#DC2626">EC 4</div>
      <div class="card-title">Inteligencia Artificial</div>
      <div class="card-desc">
        Vista integrada de todos los métodos con comparativa global.<br><br>
        <span class="badge b-red">No Supervisado</span>
        <span class="badge b-red">Clasificación</span>
        <span class="badge b-red">Regresión</span>
      </div>
      <div style="margin-top:.8rem;font-size:.8rem;color:#6B7280">
        🗃️ Todos los datasets
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Team ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="team-box">
  <strong>👥 Estudiantes</strong><br>
  <span style="color:#475569;font-size:.9rem">
    Gabriel Jesús Bello Escalona &nbsp;(A00149055) &nbsp;·&nbsp;
    Minor Castillo Loria &nbsp;(703260697)
  </span>
</div>
""", unsafe_allow_html=True)

# ── How-to ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-to">
  <strong>¿Cómo usar el framework?</strong><br>
  Usa el menú <strong>←</strong> para navegar entre casos de estudio. En cada página,
  selecciona el dataset, configura los parámetros y presiona el botón de ejecución para
  visualizar los resultados.
</div>
""", unsafe_allow_html=True)

# ── File structure expander ───────────────────────────────────────────────────
with st.expander("📁 Estructura de archivos requerida"):
    st.code("""
framework_ia/
├── Inicio.py                       ← Punto de entrada
├── pages/
│   ├── 1_EC1_No_Supervisado.py
│   ├── 2_EC2_Clasificacion.py
│   ├── 3_EC3_Regresion.py
│   └── 4_EC4_IA.py                 ← Autónomo, no requiere módulos
├── modules/
│   ├── PaqNOSup.py                 ← Copiar de Caso de Estudio 1
│   ├── GuiaClaseSupervisada.py     ← Copiar de Caso de Estudio 2
│   ├── M_Caso.py                   ← Copiar de Caso de Estudio 3
│   └── CVisualizer.py              ← Copiar de Caso de Estudio 3
├── data/
│   ├── hotel_bookings_muestra.csv  ← Caso de Estudio 1
│   ├── BankChurners.csv            ← Caso de Estudio 1
│   ├── potabilidad_V2.csv          ← Caso de Estudio 2
│   ├── diabetes_V2.csv             ← Caso de Estudio 2 / EC4
│   ├── Toyota_Price.csv            ← Caso de Estudio 3
│   └── kc_house_data.csv           ← Caso de Estudio 3
├── utils/
│   └── helpers.py
├── .streamlit/
│   └── config.toml
└── requirements.txt
    """, language="bash")
    st.markdown("**Para ejecutar:** `streamlit run Inicio.py`")