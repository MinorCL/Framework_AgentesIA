# =============================================================================
# pages/4_EC4_IA.py
# Caso de Estudio 4 — Sistema Multiagente para Machine Learning (MAS)
# EIF420O Inteligencia Artificial  |  I Ciclo 2026  |  UNA
# Dataset por defecto: diabetes_V2.csv
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

from helpers import section_header

# Sklearn — preprocesamiento
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.ensemble import GradientBoostingRegressor

# Sklearn — clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier

# Sklearn — clasificación
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Sklearn — regresión
from sklearn.linear_model import (LinearRegression, Lasso, LassoCV,
                                   Ridge, RidgeCV)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

# ===========================================================================
# CONFIGURACIÓN GLOBAL DE PÁGINA
# ===========================================================================
st.set_page_config(
    page_title="EC4 — Sistema Multiagente (IA)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.page-header {
    background: linear-gradient(90deg, #7F1D1D, #DC2626);
    color: white; padding: 1.2rem 1.8rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.page-header h2 { margin: 0; font-size: 1.5rem; font-weight: 700; }
.page-header p  { margin: .2rem 0 0 0; opacity: .8; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>🤖 EC4 — Sistema Multiagente para Machine Learning</h2>
  <p>Vista integrada: EDA · Clustering · Clasificación · Regresión · Dashboard Ejecutivo</p>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# DATA PATH (dataset por defecto del framework)
# ===========================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "diabetes_V2.csv")

# ===========================================================================
# HELPERS DE GRÁFICOS
# ===========================================================================

def fig_to_buf(fig):
    """Convierte una figura matplotlib a buffer PNG para st.image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf

# ===========================================================================
# AGENTE COORDINADOR
# ===========================================================================

class AgenteCoordinador:
    """Cerebro del sistema: carga datos y orquesta agentes."""

    def __init__(self):
        self.df_raw = None
        self.df_limpio = None
        self.target_col = None
        self.resultados = {}

    def cargar_csv(self, archivo):
        try:
            df = pd.read_csv(archivo, index_col=0)
            self.df_raw = df
            return True, df
        except Exception as e:
            return False, str(e)

    def preparar(self, df, target_col):
        """Limpieza básica y separación de target."""
        self.target_col = target_col
        df = df.dropna().drop_duplicates()
        self.df_limpio = df
        return df

    def resumen_dataset(self):
        if self.df_limpio is None:
            return {}
        df = self.df_limpio
        return {
            "Filas": df.shape[0],
            "Columnas": df.shape[1],
            "Nulos totales": int(df.isnull().sum().sum()),
            "Duplicados": int(df.duplicated().sum()),
            "Target": self.target_col,
            "Clases": df[self.target_col].unique().tolist() if self.target_col else "N/A"
        }


# ===========================================================================
# AGENTE EDA
# ===========================================================================

class AgenteEDA:
    """Análisis exploratorio de datos."""

    def __init__(self, df):
        self.df = df

    def estadisticas(self):
        return self.df.describe(include="all").T

    def valores_nulos(self):
        return self.df.isnull().sum().rename("Nulos")

    def correlacion(self):
        return self.df.corr(numeric_only=True)

    def outliers_zscore(self, umbral=3.0):
        num = self.df.select_dtypes(include="number")
        z = ((num - num.mean()) / num.std()).abs()
        return (z > umbral).sum().rename("Outliers (Z>3)")

    def plot_histogramas(self):
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        n = len(num_cols)
        cols = 3
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        axes = axes.flatten()
        palette = sns.color_palette("Set2", n)
        for i, col in enumerate(num_cols):
            axes[i].hist(self.df[col], bins=25, color=palette[i],
                         edgecolor="white", alpha=0.85)
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Histogramas", fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_boxplots(self):
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        n = len(num_cols)
        cols = 3
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        axes = axes.flatten()
        palette = sns.color_palette("Set3", n)
        for i, col in enumerate(num_cols):
            sns.boxplot(y=self.df[col], ax=axes[i], color=palette[i])
            axes[i].set_title(col, fontsize=9)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Boxplots", fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_heatmap(self):
        corr = self.correlacion()
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(corr, ax=ax, cmap=cmap, annot=True, fmt=".2f",
                    vmin=-1, vmax=1, linewidths=0.4, square=True,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Mapa de Calor – Correlaciones", fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_clase(self, target_col):
        fig, ax = plt.subplots(figsize=(6, 4))
        self.df[target_col].value_counts().plot(kind="bar", ax=ax,
                                                 color=sns.color_palette("pastel"))
        ax.set_title(f"Distribución de '{target_col}'", fontweight="bold")
        ax.set_ylabel("Frecuencia")
        ax.set_xlabel("")
        plt.tight_layout()
        return fig


# ===========================================================================
# AGENTE NO SUPERVISADO (CLUSTERING)
# ===========================================================================

class AgenteCluster:
    """Clustering: KMeans, HAC, DBSCAN con métricas de evaluación."""

    def __init__(self, df_numerico):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(df_numerico)
        self.col_names = df_numerico.columns.tolist()

    def metodo_codo(self, max_k=10):
        inercias = []
        ks = range(2, max_k + 1)
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(self.X)
            inercias.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, inercias, marker="o", color="steelblue")
        ax.set_xlabel("Número de Clusters (k)")
        ax.set_ylabel("Inercia")
        ax.set_title("Método del Codo – KMeans", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig, inercias

    def silhouette_por_k(self, max_k=10):
        scores = []
        ks = range(2, max_k + 1)
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(self.X)
            scores.append(silhouette_score(self.X, labels))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, scores, marker="s", color="darkorange")
        ax.set_xlabel("Número de Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score – KMeans", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig, scores

    def kmeans(self, k=3):
        km = KMeans(n_clusters=k, n_init=15, max_iter=500, random_state=42)
        labels = km.fit_predict(self.X)
        sil = silhouette_score(self.X, labels)
        db  = davies_bouldin_score(self.X, labels)
        ch  = calinski_harabasz_score(self.X, labels)
        # Visualización PCA 2D
        pca = PCA(n_components=2)
        comp = pca.fit_transform(self.X)
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(comp[:, 0], comp[:, 1], c=labels,
                             cmap="tab10", alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.set_title(f"KMeans (k={k})  –  PCA 2D", fontweight="bold")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        return fig, {"Silhouette": round(sil, 4),
                     "Davies-Bouldin": round(db, 4),
                     "Calinski-Harabasz": round(ch, 2)}

    def hac(self, k=3, linkage_method="ward"):
        modelo = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = modelo.fit_predict(self.X)
        sil = silhouette_score(self.X, labels)
        db  = davies_bouldin_score(self.X, labels)
        ch  = calinski_harabasz_score(self.X, labels)
        pca = PCA(n_components=2)
        comp = pca.fit_transform(self.X)
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(comp[:, 0], comp[:, 1], c=labels,
                             cmap="tab10", alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.set_title(f"HAC (k={k}, linkage={linkage_method})", fontweight="bold")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        return fig, {"Silhouette": round(sil, 4),
                     "Davies-Bouldin": round(db, 4),
                     "Calinski-Harabasz": round(ch, 2)}

    def dbscan(self, eps=0.5, min_samples=5):
        modelo = DBSCAN(eps=eps, min_samples=min_samples)
        labels = modelo.fit_predict(self.X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        ruido = int((labels == -1).sum())
        if n_clusters > 1:
            mask = labels != -1
            sil = silhouette_score(self.X[mask], labels[mask]) if mask.sum() > 1 else -1
        else:
            sil = -1
        pca = PCA(n_components=2)
        comp = pca.fit_transform(self.X)
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(comp[:, 0], comp[:, 1], c=labels,
                             cmap="tab10", alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.set_title(f"DBSCAN (eps={eps}, min_samples={min_samples})", fontweight="bold")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        return fig, {"Clusters encontrados": n_clusters,
                     "Puntos ruido": ruido,
                     "Silhouette (sin ruido)": round(sil, 4) if sil != -1 else "N/A"}

    def benchmark(self, max_k=8):
        rows = []
        for k in range(2, max_k + 1):
            # KMeans
            l = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(self.X)
            rows.append({"Algoritmo": "KMeans", "k": k,
                         "Silhouette": round(silhouette_score(self.X, l), 4),
                         "Davies-Bouldin": round(davies_bouldin_score(self.X, l), 4)})
            # HAC
            l = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(self.X)
            rows.append({"Algoritmo": "HAC-ward", "k": k,
                         "Silhouette": round(silhouette_score(self.X, l), 4),
                         "Davies-Bouldin": round(davies_bouldin_score(self.X, l), 4)})
        return pd.DataFrame(rows)


# ===========================================================================
# AGENTE SUPERVISADO – CLASIFICACIÓN
# ===========================================================================

class AgenteClasificacion:
    """Clasificación binaria sobre variable 'diabetes'."""

    def __init__(self, df, target_col):
        self.target_col = target_col
        df = df.copy()
        # Codificar categóricas
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.y_values = y.unique()
        scaler = StandardScaler()
        X_sc = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X_sc, y, test_size=0.25, random_state=42, stratify=y)

    def _metricas(self, y_true, y_pred, y_prob=None):
        mc = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        avg = "binary" if len(self.y_values) == 2 else "macro"
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)
        auc = None
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob,
                                    multi_class="ovr" if avg == "macro" else "raise")
            except Exception:
                auc = None
        return {"MC": mc, "Accuracy": round(acc, 4),
                "Precision": round(prec, 4), "Recall": round(rec, 4),
                "F1": round(f1, 4), "AUC": round(auc, 4) if auc else "N/A"}

    def _plot_mc(self, mc, titulo):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, linewidths=0.5)
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        plt.tight_layout()
        return fig

    def _entrenar(self, modelo):
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)
        y_prob = None
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(self.X_test)
            y_prob = proba[:, 1] if proba.shape[1] == 2 else proba
        return self._metricas(self.y_test, y_pred, y_prob), y_pred

    def knn(self, n_neighbors=5):
        return self._entrenar(KNeighborsClassifier(n_neighbors=n_neighbors))

    def dt(self, max_depth=5):
        return self._entrenar(DecisionTreeClassifier(max_depth=max_depth, random_state=42))

    def rf(self, n_estimators=100, max_depth=None):
        return self._entrenar(RandomForestClassifier(n_estimators=n_estimators,
                                                      max_depth=max_depth, random_state=42))

    def ada(self, n_estimators=50):
        return self._entrenar(AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=n_estimators, random_state=42))

    def xgb(self, n_estimators=100, max_depth=3):
        return self._entrenar(GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42))

    def nb(self):
        return self._entrenar(GaussianNB())

    def lr(self):
        return self._entrenar(LogisticRegression(max_iter=1000, random_state=42))

    def svm(self, C=1.0):
        return self._entrenar(SVC(C=C, probability=True, random_state=42))

    def benchmark(self):
        modelos = {
            "KNN (k=5)": self.knn,
            "Decision Tree": self.dt,
            "Random Forest": self.rf,
            "AdaBoost": self.ada,
            "GradBoost (XGB)": self.xgb,
            "Naive Bayes": self.nb,
            "Logistic Reg.": self.lr,
            "SVM": self.svm,
        }
        rows = []
        for nombre, fn in modelos.items():
            met, _ = fn()
            rows.append({"Algoritmo": nombre,
                         "Accuracy": met["Accuracy"],
                         "Precision": met["Precision"],
                         "Recall": met["Recall"],
                         "F1": met["F1"],
                         "AUC": met["AUC"]})
        return pd.DataFrame(rows).set_index("Algoritmo")

    def plot_mc(self, metricas, titulo):
        return self._plot_mc(metricas["MC"], titulo)


# ===========================================================================
# AGENTE SUPERVISADO – REGRESIÓN
# ===========================================================================

class AgenteRegresion:
    """Regresión: predice glucosa (variable continua)."""

    def __init__(self, df, target_col):
        self.target_col = target_col
        df = df.copy()
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.25, random_state=42)

    def _metricas(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
        mae  = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        return {"RMSE": round(rmse, 3), "MAE": round(mae, 3),
                "R²": round(r2, 4), "MAPE(%)": round(mape, 2)}

    def _entrenar(self, modelo):
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)
        return self._metricas(self.y_test, y_pred), y_pred

    def lineal(self):
        return self._entrenar(LinearRegression())

    def lasso(self, alpha=0.1):
        pipe = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000))
        return self._entrenar(pipe)

    def lasso_cv(self):
        pipe = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=5000, random_state=42))
        return self._entrenar(pipe)

    def ridge(self, alpha=1.0):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        return self._entrenar(pipe)

    def ridge_cv(self):
        pipe = make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-4, 4, 100)))
        return self._entrenar(pipe)

    def svr(self, kernel="rbf", C=100):
        pipe = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=0.1))
        return self._entrenar(pipe)

    def dt_reg(self, max_depth=5):
        return self._entrenar(DecisionTreeRegressor(max_depth=max_depth, random_state=42))

    def rf_reg(self, n_estimators=100):
        return self._entrenar(RandomForestRegressor(n_estimators=n_estimators, random_state=42))

    def xgb_reg(self, n_estimators=100, max_depth=4):
        return self._entrenar(GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42))

    def benchmark(self):
        modelos = {
            "Lineal Múltiple": self.lineal,
            "Lasso (α=0.1)": self.lasso,
            "LassoCV": self.lasso_cv,
            "Ridge (α=1)": self.ridge,
            "RidgeCV": self.ridge_cv,
            "SVR (rbf)": self.svr,
            "Decision Tree": self.dt_reg,
            "Random Forest": self.rf_reg,
            "GradBoost": self.xgb_reg,
        }
        rows = []
        for nombre, fn in modelos.items():
            met, _ = fn()
            rows.append({"Algoritmo": nombre, **met})
        return pd.DataFrame(rows).set_index("Algoritmo")

    def plot_pred_vs_real(self, y_pred, titulo="Predicho vs Real"):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(self.y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="k", linewidths=0.3)
        lims = [min(self.y_test.min(), y_pred.min()),
                max(self.y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
        ax.set_xlabel("Real"); ax.set_ylabel("Predicho")
        ax.set_title(titulo, fontweight="bold")
        ax.legend(); plt.tight_layout()
        return fig


# ===========================================================================
# HELPERS DE UI
# ===========================================================================

def show_metricas_clasif(met):
    cols = st.columns(5)
    cols[0].metric("Accuracy", met["Accuracy"])
    cols[1].metric("Precision", met["Precision"])
    cols[2].metric("Recall", met["Recall"])
    cols[3].metric("F1", met["F1"])
    cols[4].metric("AUC", met["AUC"])

def show_metricas_reg(met):
    cols = st.columns(4)
    cols[0].metric("RMSE", met["RMSE"])
    cols[1].metric("MAE", met["MAE"])
    cols[2].metric("R²", met["R²"])
    cols[3].metric("MAPE (%)", met["MAPE(%)"])

def highlight_best(df, ascending_cols=None, descending_cols=None):
    """Colorea mínimo (verde) para RMSE/MAE, máximo para Accuracy/F1."""
    def _highlight_min(s):
        is_min = s == s.min()
        return ["background-color: #c6efce; color: #276221" if v else "" for v in is_min]
    def _highlight_max(s):
        is_max = s == s.max()
        return ["background-color: #c6efce; color: #276221" if v else "" for v in is_max]
    styler = df.style
    if ascending_cols:
        for c in ascending_cols:
            if c in df.columns:
                styler = styler.apply(_highlight_min, subset=[c])
    if descending_cols:
        for c in descending_cols:
            if c in df.columns:
                styler = styler.apply(_highlight_max, subset=[c])
    return styler


# ===========================================================================
# FUNCIÓN PRINCIPAL
# ===========================================================================

def main():
    # ─── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🤖 MAS – ML Dashboard")
    st.sidebar.caption("UNA · Escuela de Informática")
    pagina = st.sidebar.radio("Navegación", [
        "🏠 Inicio",
        "📂 Carga de Datos",
        "🔍 EDA",
        "🔵 Clustering",
        "🟢 Clasificación",
        "🟠 Regresión",
        "🏆 Dashboard Ejecutivo",
    ])

    # ─── Estado de sesión ────────────────────────────────────────────────────
    if "coordinador" not in st.session_state:
        st.session_state.coordinador = AgenteCoordinador()
    coord = st.session_state.coordinador

    # =====================================================================
    # PÁG 1 – INICIO
    # =====================================================================
    if pagina == "🏠 Inicio":
        st.title("🤖 Sistema Multiagente para Automatización de ML")
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
**Universidad Nacional de Costa Rica**  
**Escuela de Informática**  
**EIF420O – Inteligencia Artificial | I Ciclo 2026**

---

### 🎯 Objetivo
Implementar un Sistema Multiagente (MAS) capaz de automatizar el ciclo completo
de análisis de datos y aprendizaje automático mediante un dashboard interactivo.

### 🧠 Arquitectura del Sistema
| Agente | Responsabilidad |
|---|---|
| **Coordinador** | Orquesta todos los agentes, carga y valida datos |
| **EDA** | Análisis exploratorio, estadísticas, correlaciones, outliers |
| **Clustering** | KMeans, HAC, DBSCAN con métricas Silhouette/DB/CH |
| **Clasificación** | KNN, DT, RF, AdaBoost, XGBoost, NB, LR, SVM |
| **Regresión** | Lineal, Lasso, Ridge, SVR, DT, RF, GradBoost |

### 📊 Dataset
**diabetes_V2.csv** — 16 variables clínicas para predicción de diabetes.
            """)
        with col2:
            st.info("ℹ️ Para comenzar, ve a **Carga de Datos** en el menú lateral.")
            st.success("✅ Sistema listo")

    # =====================================================================
    # PÁG 2 – CARGA DE DATOS
    # =====================================================================
    elif pagina == "📂 Carga de Datos":
        st.title("📂 Agente Coordinador – Carga de Datos")

        if os.path.exists(DEFAULT_DATASET_PATH):
            st.markdown(f"**Dataset por defecto detectado:** `{os.path.basename(DEFAULT_DATASET_PATH)}` "
                        f"({DATA_DIR})")
            usar_default = st.button("⚡ Usar dataset por defecto (diabetes_V2.csv)")
        else:
            usar_default = False
            st.warning(f"No se encontró `diabetes_V2.csv` en `{DATA_DIR}`. "
                       "Cópialo ahí o sube un CSV manualmente abajo.")

        archivo = st.file_uploader("...o sube tu propio archivo CSV", type=["csv"])

        fuente = archivo if archivo else (DEFAULT_DATASET_PATH if usar_default else None)

        if fuente is not None:
            ok, result = coord.cargar_csv(fuente)
            if ok:
                df = result
                st.success(f"✅ Archivo cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
                st.dataframe(df.head(20), use_container_width=True)

                target_col = st.selectbox("Selecciona la columna **objetivo** (target)",
                                          df.columns.tolist(),
                                          index=len(df.columns) - 1)
                if st.button("🚀 Confirmar y Preparar Dataset"):
                    coord.preparar(df, target_col)
                    st.session_state["df"] = coord.df_limpio
                    st.session_state["target"] = target_col
                    resumen = coord.resumen_dataset()
                    st.json(resumen)
                    st.success("Dataset preparado y almacenado en sesión.")
            else:
                st.error(f"Error al cargar: {result}")

        elif "df" in st.session_state:
            st.info("✅ Ya hay un dataset en sesión. Carga otro para reemplazarlo.")
            st.dataframe(st.session_state["df"].head(10), use_container_width=True)

    # =====================================================================
    # PÁG 3 – EDA
    # =====================================================================
    elif pagina == "🔍 EDA":
        st.title("🔍 Agente EDA – Análisis Exploratorio")
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset en **Carga de Datos**.")
            return
        df = st.session_state["df"]
        target = st.session_state.get("target", df.columns[-1])
        eda = AgenteEDA(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Estadísticas", "📊 Histogramas", "📦 Boxplots",
            "🌡️ Correlación", "🎯 Clase objetivo"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Descripción Estadística")
                st.dataframe(eda.estadisticas(), use_container_width=True)
            with col2:
                st.subheader("Valores Nulos")
                st.dataframe(eda.valores_nulos().to_frame(), use_container_width=True)
                st.subheader("Outliers (Z-Score > 3)")
                st.dataframe(eda.outliers_zscore().to_frame(), use_container_width=True)

        with tab2:
            st.image(fig_to_buf(eda.plot_histogramas()), use_container_width=True)

        with tab3:
            st.image(fig_to_buf(eda.plot_boxplots()), use_container_width=True)

        with tab4:
            st.image(fig_to_buf(eda.plot_heatmap()), use_container_width=True)
            st.subheader("Matriz de Correlación (numérica)")
            st.dataframe(eda.correlacion().style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                         use_container_width=True)

        with tab5:
            st.image(fig_to_buf(eda.plot_clase(target)), use_container_width=True)

    # =====================================================================
    # PÁG 4 – CLUSTERING
    # =====================================================================
    elif pagina == "🔵 Clustering":
        st.title("🔵 Agente No Supervisado – Clustering")
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset."); return
        df = st.session_state["df"]
        target = st.session_state.get("target")
        df_num = df.select_dtypes(include="number")
        df_num = df_num.drop(columns=[target] if target in df_num.columns else [])
        cluster_agent = AgenteCluster(df_num)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔍 Selección de k", "🔵 KMeans", "🌳 HAC", "🔴 DBSCAN"])

        with tab1:
            max_k = st.slider("Máximo k a explorar", 3, 15, 10)
            col1, col2 = st.columns(2)
            with col1:
                fig_codo, _ = cluster_agent.metodo_codo(max_k)
                st.image(fig_to_buf(fig_codo), use_container_width=True)
            with col2:
                fig_sil, scores = cluster_agent.silhouette_por_k(max_k)
                st.image(fig_to_buf(fig_sil), use_container_width=True)
            mejor_k = int(np.argmax(scores) + 2)
            st.success(f"📌 Mejor k por Silhouette: **{mejor_k}** (score={max(scores):.4f})")

        with tab2:
            k = st.slider("Número de clusters (KMeans)", 2, 10, 3)
            fig_km, met_km = cluster_agent.kmeans(k)
            st.image(fig_to_buf(fig_km), use_container_width=True)
            st.json(met_km)

        with tab3:
            k_hac = st.slider("Número de clusters (HAC)", 2, 10, 3, key="hac_k")
            link = st.selectbox("Linkage", ["ward", "average", "complete", "single"])
            fig_hac, met_hac = cluster_agent.hac(k_hac, link)
            st.image(fig_to_buf(fig_hac), use_container_width=True)
            st.json(met_hac)

        with tab4:
            eps = st.slider("eps", 0.1, 3.0, 0.5, 0.1)
            min_s = st.slider("min_samples", 2, 20, 5)
            fig_db, met_db = cluster_agent.dbscan(eps, min_s)
            st.image(fig_to_buf(fig_db), use_container_width=True)
            st.json(met_db)

        st.markdown("---")
        st.subheader("📊 Benchmarking de Clustering")
        with st.spinner("Calculando benchmarking..."):
            bm_df = cluster_agent.benchmark()
        st.dataframe(
            highlight_best(bm_df, ascending_cols=["Davies-Bouldin"],
                           descending_cols=["Silhouette"]),
            use_container_width=True)

    # =====================================================================
    # PÁG 5 – CLASIFICACIÓN
    # =====================================================================
    elif pagina == "🟢 Clasificación":
        st.title("🟢 Agente Supervisado – Clasificación")
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset."); return
        df = st.session_state["df"]
        target = st.session_state.get("target", df.columns[-1])
        clf_agent = AgenteClasificacion(df, target)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "KNN", "Decision Tree", "Random Forest",
            "AdaBoost", "GradBoost", "Naive Bayes",
            "Logistic Reg.", "SVM", "📊 Benchmark"
        ])

        def render_clasif(met, y_pred, titulo):
            show_metricas_clasif(met)
            st.image(fig_to_buf(clf_agent.plot_mc(met, titulo)), use_container_width=False)

        with tab1:
            k = st.slider("k vecinos", 1, 20, 5, key="knn_k")
            met, yp = clf_agent.knn(k)
            render_clasif(met, yp, f"KNN (k={k})")

        with tab2:
            depth = st.slider("max_depth", 1, 15, 5, key="dt_d")
            met, yp = clf_agent.dt(depth)
            render_clasif(met, yp, f"Decision Tree (depth={depth})")

        with tab3:
            n_est = st.slider("n_estimators", 10, 300, 100, step=10, key="rf_n")
            met, yp = clf_agent.rf(n_est)
            render_clasif(met, yp, f"Random Forest (n={n_est})")

        with tab4:
            n_ada = st.slider("n_estimators", 10, 200, 50, step=10, key="ada_n")
            met, yp = clf_agent.ada(n_ada)
            render_clasif(met, yp, f"AdaBoost (n={n_ada})")

        with tab5:
            n_xgb = st.slider("n_estimators", 10, 200, 100, step=10, key="xgb_n")
            d_xgb = st.slider("max_depth", 1, 8, 3, key="xgb_d")
            met, yp = clf_agent.xgb(n_xgb, d_xgb)
            render_clasif(met, yp, f"GradBoost (n={n_xgb}, d={d_xgb})")

        with tab6:
            met, yp = clf_agent.nb()
            render_clasif(met, yp, "Naive Bayes")

        with tab7:
            met, yp = clf_agent.lr()
            render_clasif(met, yp, "Logistic Regression")

        with tab8:
            c_svm = st.selectbox("C (regularización)", [0.1, 1.0, 10.0, 100.0], index=1, key="svm_c")
            met, yp = clf_agent.svm(c_svm)
            render_clasif(met, yp, f"SVM (C={c_svm})")

        with tab9:
            st.subheader("📊 Benchmarking Automático – Clasificación")
            with st.spinner("Entrenando todos los modelos..."):
                bm = clf_agent.benchmark()
            numeric_bm = bm.apply(pd.to_numeric, errors="coerce")
            st.dataframe(
                highlight_best(numeric_bm,
                               descending_cols=["Accuracy", "Precision", "Recall", "F1"]),
                use_container_width=True)
            mejor = numeric_bm["F1"].idxmax()
            st.success(f"🏆 Mejor modelo por F1: **{mejor}** (F1 = {numeric_bm.loc[mejor,'F1']})")

    # =====================================================================
    # PÁG 6 – REGRESIÓN
    # =====================================================================
    elif pagina == "🟠 Regresión":
        st.title("🟠 Agente Supervisado – Regresión")
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset."); return
        df = st.session_state["df"]
        num_cols = df.select_dtypes(include="number").columns.tolist()
        reg_target = st.selectbox(
            "Variable objetivo para regresión (continua):",
            num_cols, index=num_cols.index("glucosa") if "glucosa" in num_cols else 0)
        reg_agent = AgenteRegresion(df, reg_target)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Lineal", "Lasso/Ridge", "SVR", "Árbol DT", "Random Forest", "📊 Benchmark"
        ])

        def render_reg(met, yp, titulo):
            show_metricas_reg(met)
            st.image(fig_to_buf(reg_agent.plot_pred_vs_real(yp, titulo)), use_container_width=False)

        with tab1:
            met, yp = reg_agent.lineal()
            render_reg(met, yp, "Regresión Lineal Múltiple")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Lasso**")
                alpha_l = st.select_slider("Alpha Lasso", [0.001, 0.01, 0.1, 1.0, 10.0], value=0.1)
                met, yp = reg_agent.lasso(alpha_l)
                show_metricas_reg(met)
            with col2:
                st.markdown("**Ridge**")
                alpha_r = st.select_slider("Alpha Ridge", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
                met, yp = reg_agent.ridge(alpha_r)
                show_metricas_reg(met)
            st.markdown("**LassoCV y RidgeCV (alpha óptimo automático)**")
            m1, y1 = reg_agent.lasso_cv()
            m2, y2 = reg_agent.ridge_cv()
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("LassoCV"); show_metricas_reg(m1)
            with col4:
                st.markdown("RidgeCV"); show_metricas_reg(m2)

        with tab3:
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
            C_svr = st.select_slider("C", [1, 10, 50, 100, 500], value=100)
            met, yp = reg_agent.svr(kernel, C_svr)
            render_reg(met, yp, f"SVR ({kernel}, C={C_svr})")

        with tab4:
            depth_dt = st.slider("max_depth", 1, 15, 5, key="dt_reg_d")
            met, yp = reg_agent.dt_reg(depth_dt)
            render_reg(met, yp, f"Decision Tree Reg. (depth={depth_dt})")

        with tab5:
            n_rf = st.slider("n_estimators", 10, 300, 100, step=10, key="rf_reg_n")
            met, yp = reg_agent.rf_reg(n_rf)
            render_reg(met, yp, f"Random Forest Reg. (n={n_rf})")

        with tab6:
            st.subheader("📊 Benchmarking Automático – Regresión")
            with st.spinner("Entrenando todos los modelos..."):
                bm = reg_agent.benchmark()
            st.dataframe(
                highlight_best(bm, ascending_cols=["RMSE", "MAE", "MAPE(%)"],
                               descending_cols=["R²"]),
                use_container_width=True)
            mejor = bm["R²"].idxmax()
            st.success(f"🏆 Mejor modelo por R²: **{mejor}** (R² = {bm.loc[mejor,'R²']})")

    # =====================================================================
    # PÁG 7 – DASHBOARD EJECUTIVO
    # =====================================================================
    elif pagina == "🏆 Dashboard Ejecutivo":
        st.title("🏆 Dashboard Ejecutivo – Resumen MAS")
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset."); return
        df = st.session_state["df"]
        target = st.session_state.get("target", df.columns[-1])

        st.markdown("### 📋 Resumen del Dataset")
        coord.df_limpio = df
        coord.target_col = target
        resumen = coord.resumen_dataset()
        cols = st.columns(len(resumen))
        for i, (k, v) in enumerate(resumen.items()):
            cols[i].metric(k, str(v))

        st.markdown("---")
        st.markdown("### 🟢 Mejor Algoritmo – Clasificación")
        with st.spinner("Calculando benchmarking de clasificación..."):
            clf_agent = AgenteClasificacion(df, target)
            bm_clf = clf_agent.benchmark()
        numeric_bm = bm_clf.apply(pd.to_numeric, errors="coerce")
        mejor_clf = numeric_bm["F1"].idxmax()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.success(f"🏆 **{mejor_clf}**")
            show_metricas_clasif({"Accuracy": numeric_bm.loc[mejor_clf, "Accuracy"],
                                   "Precision": numeric_bm.loc[mejor_clf, "Precision"],
                                   "Recall": numeric_bm.loc[mejor_clf, "Recall"],
                                   "F1": numeric_bm.loc[mejor_clf, "F1"],
                                   "AUC": numeric_bm.loc[mejor_clf, "AUC"]})
        with col2:
            fig_bar, ax = plt.subplots(figsize=(8, 4))
            numeric_bm[["Accuracy", "F1"]].plot(kind="bar", ax=ax,
                                                  color=["steelblue", "darkorange"])
            ax.set_title("Comparación Clasificadores", fontweight="bold")
            ax.set_xticklabels(numeric_bm.index, rotation=35, ha="right")
            ax.set_ylim(0, 1); ax.legend(); plt.tight_layout()
            st.image(fig_to_buf(fig_bar), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🟠 Mejor Algoritmo – Regresión (glucosa)")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        reg_target_col = "glucosa" if "glucosa" in num_cols else num_cols[0]
        with st.spinner("Calculando benchmarking de regresión..."):
            reg_agent = AgenteRegresion(df, reg_target_col)
            bm_reg = reg_agent.benchmark()
        mejor_reg = bm_reg["R²"].idxmax()
        col3, col4 = st.columns([1, 2])
        with col3:
            st.success(f"🏆 **{mejor_reg}**")
            show_metricas_reg({"RMSE": bm_reg.loc[mejor_reg, "RMSE"],
                               "MAE": bm_reg.loc[mejor_reg, "MAE"],
                               "R²": bm_reg.loc[mejor_reg, "R²"],
                               "MAPE(%)": bm_reg.loc[mejor_reg, "MAPE(%)"]})
        with col4:
            fig_bar2, ax2 = plt.subplots(figsize=(8, 4))
            bm_reg["R²"].plot(kind="bar", ax=ax2, color="steelblue")
            ax2.set_title("Comparación Regresores (R²)", fontweight="bold")
            ax2.set_xticklabels(bm_reg.index, rotation=35, ha="right")
            ax2.set_ylabel("R²"); plt.tight_layout()
            st.image(fig_to_buf(fig_bar2), use_container_width=True)

        st.markdown("---")
        st.markdown("### 💡 Conclusiones Automáticas")
        acc_val = float(numeric_bm.loc[mejor_clf, "Accuracy"])
        r2_val  = float(bm_reg.loc[mejor_reg, "R²"])
        conclusiones = f"""
- **Dataset**: {resumen['Filas']} observaciones · {resumen['Columnas']} variables · Target: `{target}`
- **Clasificación**: El mejor clasificador fue **{mejor_clf}** con Accuracy = `{acc_val:.2%}` y F1 = `{numeric_bm.loc[mejor_clf,'F1']}`.
- **Regresión** (`{reg_target_col}`): El mejor regresor fue **{mejor_reg}** con R² = `{r2_val:.4f}`.
- {'✅ El modelo de clasificación tiene buena capacidad predictiva (Accuracy ≥ 75%).' if acc_val >= 0.75 else '⚠️ El modelo de clasificación podría mejorarse con más datos o ajuste de hiperparámetros.'}
- {'✅ El modelo de regresión explica bien la varianza (R² ≥ 0.6).' if r2_val >= 0.6 else '⚠️ El modelo de regresión tiene R² bajo; considera agregar variables o probar transformaciones.'}
        """
        st.markdown(conclusiones)


if __name__ == "__main__":
    main()